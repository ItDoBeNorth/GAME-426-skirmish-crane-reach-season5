"""S5 task 9: a unit's memory of enemies it has seen but cannot see now (the hybrid's rule part).

The teacher remembers enemies out of sight (footmen see only 4 tiles) and moves toward them when
nothing is in view; the learner could not see that, so idle archers and cavalry disagreed most.
This tracker is plain rules, not learned. Collection (imitation/experiment_tools.RecordingPlayer)
and play (agent.Agent) both update one EnemyMemory per unit from that unit's own observations,
once per activation and before encoding, so a remembered enemy means the same thing in training
and in play. It does not use chat.

A record is dropped when its tile is in sight and the enemy is not there (it moved or died), or
when it is older than MAX_AGE rounds.
"""

from sandbox.crane import me, tile, units, visible

MAX_AGE = 20  # rounds; the S5 learner's capture races run 100+ rounds, so keep it fairly long


class EnemyMemory:
    def __init__(self):
        self.records: dict[str, dict] = {}

    def update(self, observation):
        """Refresh from what this unit sees now, then forget what sight proves gone or what is too old."""
        state = observation["observation"]
        round_now = state["round"]
        here = me.position(observation)
        own_type = state["self"]["type"]
        on_hill = tile.terrain_at(observation, here)["terrain"] == "hill"
        vision = units.STATS[own_type].vision + on_hill
        seen = set()
        for enemy in visible.enemies(observation):
            seen.add(enemy["unit_id"])
            self.records[enemy["unit_id"]] = {
                "type": enemy["type"],
                "position": dict(enemy["position"]),
                "hit_points": enemy["hit_points"],
                "round": round_now,
            }
        for unit_id, record in list(self.records.items()):
            if unit_id in seen:
                continue
            if round_now - record["round"] > MAX_AGE or tile.distance(here, record["position"]) <= vision:
                del self.records[unit_id]

    def remembered(self, observation):
        """Records of enemies not in sight now, freshest first, each with its age in rounds."""
        round_now = observation["observation"]["round"]
        seen = {enemy["unit_id"] for enemy in visible.enemies(observation)}
        out = [
            {**record, "age": round_now - record["round"]}
            for unit_id, record in self.records.items()
            if unit_id not in seen
        ]
        return sorted(out, key=lambda record: (record["age"], record["position"]["q"], record["position"]["r"]))
