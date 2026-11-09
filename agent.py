"""Choose a reachable tile, then a target, with the trained PyTorch policy.

We encode the acting unit, visible units, and zones once. Movement candidates read that
context through attention. Each target candidate points directly at its enemy's token.
Both decisions use legal options, with the first option breaking equal-score ties.
"""

from pathlib import Path

import torch
from encoder import (
    USE_MEMORY_FEATURES,
    describe_move,
    describe_target,
    encode_entities,
    move_options,
    target_entity_indices,
    target_options,
)
from enemy_memory import EnemyMemory
from imitation_model import load_policy
from sandbox.crane import me, tile, visible

# S5 task 9 (hybrid rule part): remember enemies out of sight and show them to the model. The
# model must be trained with the same setting, so this follows the encoder's switch (off: the memory
# learner lost to the s5-features-b learner; scores in encoder.USE_MEMORY_FEATURES).
USE_MEMORY = USE_MEMORY_FEATURES
# S5 learner fixes (hybrid rules on top of the network's tile scores). Measured on the t10 learner
# against Banner: idle archers stood still on 26% of their activations (teacher 4%), and footmen
# stood on a zone under archer fire 11.7 times per game (teacher 1.7). Each rule only replaces a
# "stay" pick, with the network's best-scoring tile among the allowed ones.
# Seeds 600-619 wins/40 (mean) for the t10 model: plain = Banner 40 (97.9), teacher 23 (54.6),
# Banner learner 40 (94.2); both rules = 40 (98.7), 26 (63.6), 40 (96.4), 21/40 vs plain t10;
# unstick only = 40 (98.7), 24 (58.3), 40 (98.3); answer only = 40 (97.9), 25 (58.5), 40 (95.7).
# UNSTICK_IDLE_ARCHERS: an archer with no enemy in sight that also stayed last activation moves.
# Idle archers stood still 8% of the time instead of 26%.
UNSTICK_IDLE_ARCHERS = True
# FOOTMEN_ANSWER_ARCHERS: a footman with no enemy adjacent, shot at by a visible enemy archer it
# cannot reach, takes a tile closer to that archer. It rarely fires: footmen see 4 tiles and
# archers shoot from 6, so the shooter is usually out of sight. Kept on: "both" scored best.
FOOTMEN_ANSWER_ARCHERS = True
# S5 task 11, DAgger (tools/dagger_round.sh, model runs/t11.pt): one round of learner-driven
# corrections lost to t10 (head-to-head 18/40; teacher 45.0 vs 54.6). The teacher copies label
# idle archers "stay" twice as often when learners play (15% vs 7.5%), because the teacher keeps
# archers behind its footmen and the learner's footmen are more passive. Not installed.
ARCHER_RANGE = 6


def option_tensor(rows):
    """Add a batch dimension: one decision, several rows, several features per row."""
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(0)


@torch.inference_mode()
def choose_action(model, observation, remembered=(), override=None):
    """S5: override(observation, moves, scores) may return another move index (hybrid rules)."""
    # Read the battlefield once. Both scorers use this same context.
    # S5 task 9: remembered enemies (enemy_memory) join the visible units as their own tokens.
    entities, enemy_indices = encode_entities(observation, remembered)
    # A mask marks real rows. During play every row is real; training also has padding.
    entity_mask = torch.ones((1, len(entities)), dtype=torch.bool)
    context = model.encode(option_tensor(entities), entity_mask)

    moves = move_options(observation)
    targets = target_options(observation)
    if not moves or not targets:
        raise ValueError("The observation must contain legal movement and target options")
    # Score destinations, keeping one legal path to each. argmax keeps the first tie.
    move_rows = option_tensor([describe_move(observation, end, path, remembered) for path, end in moves])
    move_mask = torch.ones((1, len(moves)), dtype=torch.bool)
    scores = model.score_moves(context, entity_mask, move_rows, move_mask)
    chosen = int(scores[0].argmax())
    if override is not None:
        replaced = override(observation, moves, scores[0].tolist(), chosen)
        chosen = chosen if replaced is None else replaced
    path, end = moves[chosen]

    # Describe targets from the tile we actually chose. The pointers connect target IDs
    # to the enemy tokens in context; -1 selects the model's learned no-target value.
    target_rows = option_tensor([describe_target(observation, end, target) for target in targets])
    pointers = torch.tensor([target_entity_indices(observation, targets, enemy_indices)])
    target_mask = torch.ones((1, len(targets)), dtype=torch.bool)
    scores = model.score_targets(context, target_rows, pointers, target_mask)
    return {"path": path, "target": targets[int(scores[0].argmax())]}


class Agent:
    def __init__(self):
        # A model this small runs fastest on one CPU thread. The setting applies to the whole
        # process, so we set it before the first inference.
        torch.set_num_threads(1)
        self.model = load_policy(Path(__file__).with_name("imitation-policy.pt"))
        self.memory = EnemyMemory()  # S5 task 9; reset() starts a fresh one each match

    def reset(self, seed, observation):
        # This learner carries no private memory between activations. Its weights stay fixed.
        # S5 task 9 (hybrid): it now keeps one rule-based memory of enemies out of sight
        # (enemy_memory.py), the same tracker collection used. USE_MEMORY switches it off.
        self.memory = EnemyMemory()
        self.stayed = False  # S5: did our previous activation end where it started?

    def act(self, observation):
        here = me.position(observation)
        remembered = ()
        if USE_MEMORY:
            self.memory.update(observation)
            remembered = self.memory.remembered(observation)
        action = choose_action(self.model, observation, remembered, override=self.override)
        self.stayed = tile.at_path_end(here, action["path"]) == here
        return action

    def override(self, observation, moves, scores, chosen):
        """S5 learner fixes: replace some "stay" picks (see the switches at the top). None = keep."""
        here = me.position(observation)
        if moves[chosen][1] != here:
            return None
        unit_type = me.unit_type(observation)
        enemies = visible.enemies(observation)

        def best(allowed):
            options = [index for index, (_, end) in enumerate(moves) if allowed(end)]
            return max(options, key=lambda index: scores[index]) if options else None

        if UNSTICK_IDLE_ARCHERS and unit_type == "archer" and not enemies and getattr(self, "stayed", False):
            return best(lambda end: end != here)
        if FOOTMEN_ANSWER_ARCHERS and unit_type == "footman" and enemies:
            if any(tile.distance(here, enemy["position"]) <= 1 for enemy in enemies):
                return None
            shooters = [
                enemy
                for enemy in enemies
                if enemy["type"] == "archer" and tile.distance(here, enemy["position"]) <= ARCHER_RANGE
            ]
            if shooters:
                archer = min(shooters, key=lambda enemy: tile.distance(here, enemy["position"]))
                gap = tile.distance(here, archer["position"])
                return best(lambda end: tile.distance(end, archer["position"]) < gap)
        return None
