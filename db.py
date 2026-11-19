"""Módulo de acesso à Base de Conhecimento SQLite para vídeos virais."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "base_conhecimento.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Abre uma conexão SQLite com row_factory configurado para dicionário."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Inicializa as tabelas da base de conhecimento se ainda não existirem."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS videos_referencia (
                shortcode TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                uploader TEXT,
                caption TEXT,
                categoria_fisica TEXT,
                descricao_visual TEXT,
                likes INTEGER,
                comments INTEGER,
                duration_sec REAL,
                width INTEGER,
                height INTEGER,
                fps REAL,
                video_path TEXT,
                audio_raw_path TEXT,
                audio_asmr_path TEXT,
                frames_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS roteiros_ia (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ref_shortcode TEXT NOT NULL,
                variacao_tema TEXT,
                wan21_positive_prompt TEXT,
                wan21_negative_prompt TEXT,
                recommended_denoise REAL,
                titulo_viral_pt TEXT,
                titulo_viral_en TEXT,
                hashtags TEXT,
                notas_fisica_som TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (ref_shortcode) REFERENCES videos_referencia(shortcode)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS publicacoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ref_shortcode TEXT NOT NULL,
                plataforma TEXT NOT NULL,
                video_final_path TEXT NOT NULL,
                titulo TEXT,
                legenda_completa TEXT,
                hashtags TEXT,
                primeiro_comentario TEXT,
                status TEXT NOT NULL,
                post_url TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (ref_shortcode) REFERENCES videos_referencia(shortcode)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS modelos_locais (
                model_key TEXT PRIMARY KEY,
                nome TEXT NOT NULL,
                provider TEXT NOT NULL,
                repo_id TEXT,
                filename TEXT,
                target_dir TEXT,
                local_path TEXT,
                size_mb REAL,
                obrigatorio INTEGER DEFAULT 1,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pipeline_execucoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                flow_name TEXT NOT NULL,
                etapa TEXT NOT NULL,
                shortcode TEXT,
                status TEXT NOT NULL,
                detalhes TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def upsert_video_referencia(data: dict[str, Any], db_path: Path = DB_PATH) -> None:
    """Insere ou atualiza os metadados e caminhos locais de um vídeo de referência."""
    init_db(db_path)
    frames_val = data.get("frames_json")
    if isinstance(frames_val, (list, dict)):
        frames_val = json.dumps(frames_val, ensure_ascii=False)

    created_at = data.get("created_at") or _utc_now_iso()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO videos_referencia (
                shortcode, url, uploader, caption, categoria_fisica,
                descricao_visual, likes, comments, duration_sec,
                width, height, fps, video_path, audio_raw_path,
                audio_asmr_path, frames_json, created_at
            ) VALUES (
                :shortcode, :url, :uploader, :caption, :categoria_fisica,
                :descricao_visual, :likes, :comments, :duration_sec,
                :width, :height, :fps, :video_path, :audio_raw_path,
                :audio_asmr_path, :frames_json, :created_at
            )
            ON CONFLICT(shortcode) DO UPDATE SET
                url = excluded.url,
                uploader = COALESCE(excluded.uploader, videos_referencia.uploader),
                caption = COALESCE(excluded.caption, videos_referencia.caption),
                categoria_fisica = COALESCE(excluded.categoria_fisica, videos_referencia.categoria_fisica),
                descricao_visual = COALESCE(excluded.descricao_visual, videos_referencia.descricao_visual),
                likes = COALESCE(excluded.likes, videos_referencia.likes),
                comments = COALESCE(excluded.comments, videos_referencia.comments),
                duration_sec = COALESCE(excluded.duration_sec, videos_referencia.duration_sec),
                width = COALESCE(excluded.width, videos_referencia.width),
                height = COALESCE(excluded.height, videos_referencia.height),
                fps = COALESCE(excluded.fps, videos_referencia.fps),
                video_path = COALESCE(excluded.video_path, videos_referencia.video_path),
                audio_raw_path = COALESCE(excluded.audio_raw_path, videos_referencia.audio_raw_path),
                audio_asmr_path = COALESCE(excluded.audio_asmr_path, videos_referencia.audio_asmr_path),
                frames_json = COALESCE(excluded.frames_json, videos_referencia.frames_json)
            """,
            {
                "shortcode": data["shortcode"],
                "url": data["url"],
                "uploader": data.get("uploader"),
                "caption": data.get("caption"),
                "categoria_fisica": data.get("categoria_fisica"),
                "descricao_visual": data.get("descricao_visual"),
                "likes": data.get("likes"),
                "comments": data.get("comments"),
                "duration_sec": data.get("duration_sec"),
                "width": data.get("width"),
                "height": data.get("height"),
                "fps": data.get("fps"),
                "video_path": data.get("video_path"),
                "audio_raw_path": data.get("audio_raw_path"),
                "audio_asmr_path": data.get("audio_asmr_path"),
                "frames_json": frames_val,
                "created_at": created_at,
            },
        )
        conn.commit()


def insert_roteiro_ia(
    data: dict[str, Any],
    replace_existing: bool = False,
    db_path: Path = DB_PATH,
) -> int:
    """Insere um roteiro e prompt Wan 2.1 gerado pelo Llama na base de conhecimento."""
    init_db(db_path)
    created_at = data.get("created_at") or _utc_now_iso()
    hashtags = data.get("hashtags", "")
    if isinstance(hashtags, list):
        hashtags = " ".join(str(tag) for tag in hashtags)

    with get_connection(db_path) as conn:
        if replace_existing:
            conn.execute(
                "DELETE FROM roteiros_ia WHERE ref_shortcode = ?",
                (data["ref_shortcode"],),
            )
        cur = conn.execute(
            """
            INSERT INTO roteiros_ia (
                ref_shortcode, variacao_tema, wan21_positive_prompt,
                wan21_negative_prompt, recommended_denoise, titulo_viral_pt,
                titulo_viral_en, hashtags, notas_fisica_som, created_at
            ) VALUES (
                :ref_shortcode, :variacao_tema, :wan21_positive_prompt,
                :wan21_negative_prompt, :recommended_denoise, :titulo_viral_pt,
                :titulo_viral_en, :hashtags, :notas_fisica_som, :created_at
            )
            """,
            {
                "ref_shortcode": data["ref_shortcode"],
                "variacao_tema": data.get("variacao_tema"),
                "wan21_positive_prompt": data.get("wan21_positive_prompt"),
                "wan21_negative_prompt": data.get("wan21_negative_prompt"),
                "recommended_denoise": data.get("recommended_denoise"),
                "titulo_viral_pt": data.get("titulo_viral_pt"),
                "titulo_viral_en": data.get("titulo_viral_en"),
                "hashtags": hashtags,
                "notas_fisica_som": data.get("notas_fisica_som"),
                "created_at": created_at,
            },
        )
        conn.commit()
        return int(cur.lastrowid)


def get_all_videos(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Retorna todos os vídeos de referência catalogados."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM videos_referencia ORDER BY created_at ASC, shortcode ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_video_by_shortcode(
    shortcode: str, db_path: Path = DB_PATH
) -> dict[str, Any] | None:
    """Busca um vídeo específico pelo shortcode."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM videos_referencia WHERE shortcode = ?",
            (shortcode,),
        ).fetchone()
    return dict(row) if row else None


def get_videos_without_roteiro(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Retorna vídeos de referência que ainda não possuem roteiro gerado na tabela roteiros_ia."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT v.*
            FROM videos_referencia v
            LEFT JOIN roteiros_ia r ON r.ref_shortcode = v.shortcode
            WHERE r.id IS NULL
            ORDER BY v.created_at ASC, v.shortcode ASC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_full_catalog(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Retorna todos os vídeos com seus respectivos roteiros mais recentes associados."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                v.*,
                r.id AS roteiro_id,
                r.variacao_tema,
                r.wan21_positive_prompt,
                r.wan21_negative_prompt,
                r.recommended_denoise,
                r.titulo_viral_pt,
                r.titulo_viral_en,
                r.hashtags,
                r.notas_fisica_som,
                r.created_at AS roteiro_created_at
            FROM videos_referencia v
            LEFT JOIN roteiros_ia r
                ON r.id = (
                    SELECT r2.id
                    FROM roteiros_ia r2
                    WHERE r2.ref_shortcode = v.shortcode
                    ORDER BY r2.id DESC
                    LIMIT 1
                )
            ORDER BY v.created_at ASC, v.shortcode ASC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def upsert_publicacao(data: dict[str, Any], db_path: Path = DB_PATH) -> int:
    """Insere ou atualiza o registro de publicação de um vídeo em uma plataforma."""
    init_db(db_path)
    created_at = data.get("created_at") or _utc_now_iso()
    hashtags = data.get("hashtags", "")
    if isinstance(hashtags, list):
        hashtags = " ".join(str(tag) for tag in hashtags)

    params = {
        "ref_shortcode": data["ref_shortcode"],
        "plataforma": data["plataforma"],
        "video_final_path": data["video_final_path"],
        "titulo": data.get("titulo"),
        "legenda_completa": data.get("legenda_completa"),
        "hashtags": hashtags,
        "primeiro_comentario": data.get("primeiro_comentario"),
        "status": data.get("status", "pronto"),
        "post_url": data.get("post_url"),
        "created_at": created_at,
    }

    with get_connection(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM publicacoes WHERE ref_shortcode = ? AND plataforma = ?",
            (params["ref_shortcode"], params["plataforma"]),
        ).fetchone()
        if existing:
            pub_id = int(existing["id"])
            conn.execute(
                """
                UPDATE publicacoes SET
                    video_final_path = :video_final_path,
                    titulo = :titulo,
                    legenda_completa = :legenda_completa,
                    hashtags = :hashtags,
                    primeiro_comentario = :primeiro_comentario,
                    status = :status,
                    post_url = COALESCE(:post_url, post_url),
                    created_at = :created_at
                WHERE id = :id
                """,
                {**params, "id": pub_id},
            )
            conn.commit()
            return pub_id

        cur = conn.execute(
            """
            INSERT INTO publicacoes (
                ref_shortcode, plataforma, video_final_path, titulo,
                legenda_completa, hashtags, primeiro_comentario,
                status, post_url, created_at
            ) VALUES (
                :ref_shortcode, :plataforma, :video_final_path, :titulo,
                :legenda_completa, :hashtags, :primeiro_comentario,
                :status, :post_url, :created_at
            )
            """,
            params,
        )
        conn.commit()
        return int(cur.lastrowid)


def get_publicacoes(
    shortcodes: list[str] | None = None, db_path: Path = DB_PATH
) -> list[dict[str, Any]]:
    """Retorna registros de publicações, opcionalmente filtrados por shortcodes."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        if shortcodes:
            placeholders = ",".join("?" for _ in shortcodes)
            rows = conn.execute(
                f"SELECT * FROM publicacoes WHERE ref_shortcode IN ({placeholders}) ORDER BY id ASC",
                tuple(shortcodes),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM publicacoes ORDER BY id ASC"
            ).fetchall()
    return [dict(r) for r in rows]


def upsert_modelo_local(data: dict[str, Any], db_path: Path = DB_PATH) -> None:
    """Insere ou atualiza o estado local de um modelo ou runtime de IA."""
    init_db(db_path)
    updated_at = data.get("updated_at") or _utc_now_iso()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO modelos_locais (
                model_key, nome, provider, repo_id, filename,
                target_dir, local_path, size_mb, obrigatorio,
                status, updated_at
            ) VALUES (
                :model_key, :nome, :provider, :repo_id, :filename,
                :target_dir, :local_path, :size_mb, :obrigatorio,
                :status, :updated_at
            )
            ON CONFLICT(model_key) DO UPDATE SET
                nome = excluded.nome,
                provider = excluded.provider,
                repo_id = excluded.repo_id,
                filename = excluded.filename,
                target_dir = excluded.target_dir,
                local_path = excluded.local_path,
                size_mb = excluded.size_mb,
                obrigatorio = excluded.obrigatorio,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            {
                "model_key": data["model_key"],
                "nome": data["nome"],
                "provider": data["provider"],
                "repo_id": data.get("repo_id"),
                "filename": data.get("filename"),
                "target_dir": data.get("target_dir"),
                "local_path": data.get("local_path"),
                "size_mb": data.get("size_mb"),
                "obrigatorio": int(data.get("obrigatorio", 1)),
                "status": data["status"],
                "updated_at": updated_at,
            },
        )
        conn.commit()


def get_modelos_locais(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Lista todos os modelos locais registrados no banco."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM modelos_locais ORDER BY obrigatorio DESC, model_key ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def log_pipeline_step(
    run_id: str,
    flow_name: str,
    etapa: str,
    status: str,
    detalhes: str = "",
    shortcode: str | None = None,
    db_path: Path = DB_PATH,
) -> int:
    """Registra uma etapa executada pelo orquestrador Prefect na tabela pipeline_execucoes."""
    init_db(db_path)
    created_at = _utc_now_iso()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO pipeline_execucoes (
                run_id, flow_name, etapa, shortcode, status, detalhes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, flow_name, etapa, shortcode, status, detalhes, created_at),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_pipeline_execucoes(
    limit: int = 50, db_path: Path = DB_PATH
) -> list[dict[str, Any]]:
    """Retorna os últimos registros de execução da linha de montagem."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM pipeline_execucoes ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]
