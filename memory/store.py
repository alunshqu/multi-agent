import sqlite3
import json
from pathlib import Path
from memory.layers import Pattern, Procedure, Episode, Skill, EvolutionEntry
import config

_BASE = Path.home() / ".multi_agent"
_DB_PATH = _BASE / "memory.db"


def _init_chroma_client():
    """连接到本地 ChromaDB HTTP 服务，失败时返回 None。"""
    try:
        import chromadb
        client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
        client.heartbeat()  # 验证连接
        col = client.get_or_create_collection("episodes")
        return col
    except Exception:
        return None


class MemoryStore:
    def __init__(self):
        _BASE.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._col = _init_chroma_client()  # 连接已运行的服务，无需下载模型

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                trigger_keywords TEXT DEFAULT '[]',
                confidence REAL DEFAULT 0.5,
                source_episodes TEXT DEFAULT '[]',
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS procedures (
                id TEXT PRIMARY KEY,
                system_name TEXT NOT NULL,
                description TEXT DEFAULT '',
                content TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_used TEXT,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                intent TEXT NOT NULL,
                agents_used TEXT DEFAULT '[]',
                task_shape TEXT DEFAULT 'single',
                systems TEXT DEFAULT '[]',
                outcome TEXT NOT NULL,
                user_feedback TEXT DEFAULT 'pending',
                failure_reason TEXT,
                execution_summary TEXT DEFAULT '',
                created_at TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
                USING fts5(id UNINDEXED, intent, content=episodes, content_rowid=rowid);
            CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                workflow TEXT NOT NULL,
                trigger_patterns TEXT DEFAULT '[]',
                typical_agents TEXT DEFAULT '[]',
                typical_systems TEXT DEFAULT '[]',
                parameters TEXT DEFAULT '{}',
                version INTEGER DEFAULT 1,
                usage_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                source_episodes TEXT DEFAULT '[]',
                evolution_log TEXT DEFAULT '[]',
                created_at TEXT,
                updated_at TEXT
            );
        """)
        self._conn.commit()

    # ── Pattern ───────────────────────────────────────────────────────────────

    def save_pattern(self, p: Pattern):
        self._conn.execute(
            "INSERT OR REPLACE INTO patterns VALUES (?,?,?,?,?,?,?)",
            (p.id, p.content,
             json.dumps(p.trigger_keywords, ensure_ascii=False),
             p.confidence,
             json.dumps(p.source_episodes),
             p.created_at, p.updated_at),
        )
        self._conn.commit()

    def load_all_patterns(self) -> list[Pattern]:
        rows = self._conn.execute(
            "SELECT * FROM patterns ORDER BY confidence DESC"
        ).fetchall()
        return [self._to_pattern(r) for r in rows]

    # ── Procedure ─────────────────────────────────────────────────────────────

    def save_procedure(self, p: Procedure):
        self._conn.execute(
            "INSERT OR REPLACE INTO procedures VALUES (?,?,?,?,?,?,?,?)",
            (p.id, p.system_name, p.description, p.content,
             p.success_count, p.failure_count, p.last_used, p.created_at),
        )
        self._conn.commit()

    def find_procedures(self, systems: list[str]) -> list[Procedure]:
        if not systems:
            return []
        ph = ",".join("?" * len(systems))
        rows = self._conn.execute(
            f"SELECT * FROM procedures WHERE system_name IN ({ph})", systems
        ).fetchall()
        return [self._to_procedure(r) for r in rows]

    def procedure_exists(self, system: str) -> bool:
        return self._conn.execute(
            "SELECT id FROM procedures WHERE system_name=?", (system,)
        ).fetchone() is not None

    # ── Episode ───────────────────────────────────────────────────────────────

    def save_episode(self, e: Episode):
        self._conn.execute(
            "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?)",
            (e.id, e.intent,
             json.dumps(e.agents_used),
             e.task_shape,
             json.dumps(e.systems, ensure_ascii=False),
             e.outcome, e.user_feedback,
             e.failure_reason, e.execution_summary,
             e.created_at),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO episodes_fts(id, intent) VALUES (?,?)",
            (e.id, e.intent),
        )
        self._conn.commit()

        if self._col is not None:
            try:
                self._col.upsert(
                    ids=[e.id],
                    documents=[e.intent],
                    metadatas=[{
                        "outcome": e.outcome,
                        "task_shape": e.task_shape,
                        "systems": ",".join(e.systems),
                        "agents": ",".join(e.agents_used),
                    }],
                )
            except Exception:
                pass

    def update_feedback(self, episode_id: str, feedback: str):
        self._conn.execute(
            "UPDATE episodes SET user_feedback=? WHERE id=?",
            (feedback, episode_id),
        )
        self._conn.commit()

    def find_similar_episodes(
        self, intent: str, systems: list[str], top_k: int = 3
    ) -> list[Episode]:
        # 优先用 ChromaDB 语义检索
        if self._col is not None:
            return self._find_by_chroma(intent, systems, top_k)
        # 降级：SQLite FTS5 关键词检索
        return self._find_by_fts(intent, systems, top_k)

    def _find_by_chroma(self, intent: str, systems: list[str], top_k: int) -> list[Episode]:
        try:
            results = self._col.query(query_texts=[intent], n_results=min(top_k * 3, 20))
            if not results["ids"][0]:
                return []
            ranked, fallback = [], []
            for i, meta in enumerate(results["metadatas"][0]):
                eid = results["ids"][0][i]
                ep_systems = set(meta.get("systems", "").split(","))
                if systems and ep_systems.intersection(systems):
                    ranked.append(eid)
                else:
                    fallback.append(eid)
            candidate_ids = (ranked + fallback)[:top_k]
            if not candidate_ids:
                return []
            ph = ",".join("?" * len(candidate_ids))
            rows = self._conn.execute(
                f"SELECT * FROM episodes WHERE id IN ({ph})", candidate_ids
            ).fetchall()
            return [self._to_episode(r) for r in rows]
        except Exception:
            return self._find_by_fts(intent, systems, top_k)

    def _find_by_fts(self, intent: str, systems: list[str], top_k: int) -> list[Episode]:
        if not intent.strip():
            return []
        keywords = " OR ".join(f'"{w}"' for w in intent.split() if len(w) > 1)
        if not keywords:
            return []
        try:
            rows = self._conn.execute(
                """SELECT e.* FROM episodes e
                   JOIN episodes_fts f ON e.id = f.id
                   WHERE episodes_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (keywords, top_k * 3),
            ).fetchall()
        except Exception:
            return []
        episodes = [self._to_episode(r) for r in rows]
        if systems:
            with_overlap = [e for e in episodes if set(e.systems).intersection(systems)]
            without = [e for e in episodes if e not in with_overlap]
            return (with_overlap + without)[:top_k]
        return episodes[:top_k]

    def find_success_episodes(self, agents: list[str], systems: list[str]) -> list[Episode]:
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE outcome='success'"
        ).fetchall()
        episodes = [self._to_episode(r) for r in rows]
        result = []
        for e in episodes:
            if set(e.agents_used).intersection(agents):
                if not systems or set(e.systems).intersection(systems):
                    result.append(e)
        return result

    def find_failure_episodes(self, systems: list[str]) -> list[Episode]:
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE outcome='failure' AND failure_reason IS NOT NULL"
        ).fetchall()
        episodes = [self._to_episode(r) for r in rows]
        if systems:
            episodes = [e for e in episodes if set(e.systems).intersection(systems)]
        return episodes

    # ── Skill ─────────────────────────────────────────────────────────────────

    def save_skill(self, s: Skill):
        evol_raw = json.dumps(
            [{"version": e.version, "reason": e.reason, "changed_at": e.changed_at}
             for e in s.evolution_log],
            ensure_ascii=False,
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO skills VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (s.id, s.name, s.description, s.workflow,
             json.dumps(s.trigger_patterns, ensure_ascii=False),
             json.dumps(s.typical_agents),
             json.dumps(s.typical_systems, ensure_ascii=False),
             json.dumps(s.parameters, ensure_ascii=False),
             s.version, s.usage_count, s.success_count,
             json.dumps(s.source_episodes),
             evol_raw, s.created_at, s.updated_at),
        )
        self._conn.commit()

    def load_all_skills(self) -> list[Skill]:
        rows = self._conn.execute(
            "SELECT * FROM skills ORDER BY usage_count DESC"
        ).fetchall()
        return [self._to_skill(r) for r in rows]

    def find_matching_skills(self, intent: str, agents: list[str], systems: list[str]) -> list[Skill]:
        all_skills = self.load_all_skills()
        scored: list[tuple[int, Skill]] = []
        intent_lower = intent.lower()
        for s in all_skills:
            score = 0
            score += len(set(s.typical_agents).intersection(agents)) * 3
            score += len(set(s.typical_systems).intersection(systems)) * 2
            score += sum(1 for kw in s.trigger_patterns if kw.lower() in intent_lower)
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:2]]

    def increment_skill_usage(self, skill_id: str, success: bool):
        self._conn.execute(
            "UPDATE skills SET usage_count=usage_count+1, success_count=success_count+? WHERE id=?",
            (1 if success else 0, skill_id),
        )
        self._conn.commit()

    def skill_covers_episodes(self, episode_ids: list[str]) -> bool:
        ep_set = set(episode_ids)
        for s in self.load_all_skills():
            if ep_set.issubset(set(s.source_episodes)):
                return True
        return False

    # ── Converters ────────────────────────────────────────────────────────────

    def _to_pattern(self, r) -> Pattern:
        return Pattern(
            id=r["id"], content=r["content"],
            trigger_keywords=json.loads(r["trigger_keywords"]),
            confidence=r["confidence"],
            source_episodes=json.loads(r["source_episodes"]),
            created_at=r["created_at"], updated_at=r["updated_at"],
        )

    def _to_procedure(self, r) -> Procedure:
        return Procedure(
            id=r["id"], system_name=r["system_name"],
            description=r["description"], content=r["content"],
            success_count=r["success_count"], failure_count=r["failure_count"],
            last_used=r["last_used"], created_at=r["created_at"],
        )

    def _to_episode(self, r) -> Episode:
        return Episode(
            id=r["id"], intent=r["intent"],
            agents_used=json.loads(r["agents_used"]),
            task_shape=r["task_shape"],
            systems=json.loads(r["systems"]),
            outcome=r["outcome"], user_feedback=r["user_feedback"],
            failure_reason=r["failure_reason"],
            execution_summary=r["execution_summary"] or "",
            created_at=r["created_at"],
        )

    def _to_skill(self, r) -> Skill:
        evol_raw = json.loads(r["evolution_log"])
        evol_log = [
            EvolutionEntry(version=e["version"], reason=e["reason"], changed_at=e["changed_at"])
            for e in evol_raw
        ]
        return Skill(
            id=r["id"], name=r["name"], description=r["description"],
            workflow=r["workflow"],
            trigger_patterns=json.loads(r["trigger_patterns"]),
            typical_agents=json.loads(r["typical_agents"]),
            typical_systems=json.loads(r["typical_systems"]),
            parameters=json.loads(r["parameters"]),
            version=r["version"], usage_count=r["usage_count"],
            success_count=r["success_count"],
            source_episodes=json.loads(r["source_episodes"]),
            evolution_log=evol_log,
            created_at=r["created_at"], updated_at=r["updated_at"],
        )
