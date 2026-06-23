import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


TABLES = {
    "image_chunks": {
        "id": "id",
        "embedding": "embedding",
        "label_columns": ["image_path", "bbox", "text_chunk_id"],
    },
    "text_chunks": {
        "id": "id",
        "embedding": "embedding",
        "label_columns": ["content"],
    },
}


@dataclass
class JdbcInfo:
    host: str
    port: str
    database: str


def parse_jdbc_url(url: str) -> JdbcInfo:
    match = re.match(r"^jdbc:postgresql://([^:/]+)(?::(\d+))?/([^?]+)", url)
    if not match:
        raise ValueError(f"Unsupported JDBC URL: {url}")
    return JdbcInfo(
        host=match.group(1),
        port=match.group(2) or "5432",
        database=match.group(3),
    )


def run_command(args: List[str], env: Optional[Dict[str, str]] = None) -> str:
    proc = subprocess.run(
        args,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        cmd = " ".join(args)
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}\n{proc.stderr.strip()}")
    return proc.stdout


def autodetect_container(port: str) -> Optional[str]:
    if shutil.which("docker") is None:
        return None
    try:
        out = run_command(["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}"])
    except Exception:
        return None

    needle = f":{port}->5432/tcp"
    for line in out.splitlines():
        if needle in line or f"0.0.0.0:{port}->5432/tcp" in line:
            return line.split("\t", 1)[0]
    return None


class PsqlRunner:
    def __init__(
        self,
        jdbc: str,
        user: str,
        password: Optional[str],
        docker_container: Optional[str],
    ) -> None:
        self.jdbc = parse_jdbc_url(jdbc)
        self.user = user
        self.password = password
        self.docker_container = docker_container

    def base_args(self) -> List[str]:
        if self.docker_container:
            return [
                "docker",
                "exec",
                self.docker_container,
                "psql",
                "-U",
                self.user,
                "-d",
                self.jdbc.database,
                "-X",
                "-v",
                "ON_ERROR_STOP=1",
                "-P",
                "pager=off",
            ]

        return [
            "psql",
            "-h",
            self.jdbc.host,
            "-p",
            self.jdbc.port,
            "-U",
            self.user,
            "-d",
            self.jdbc.database,
            "-X",
            "-v",
            "ON_ERROR_STOP=1",
            "-P",
            "pager=off",
        ]

    def query_csv(self, sql: str) -> List[Dict[str, str]]:
        copy_sql = f"COPY ({sql}) TO STDOUT WITH CSV HEADER"
        env = os.environ.copy()
        if self.password and not self.docker_container:
            env["PGPASSWORD"] = self.password
        out = run_command(self.base_args() + ["-c", copy_sql], env=env)
        return list(csv.DictReader(out.splitlines()))


def as_number(value: str) -> Any:
    if value is None or value == "":
        return None
    if value in {"t", "true", "True"}:
        return True
    if value in {"f", "false", "False"}:
        return False
    try:
        if re.match(r"^-?\d+$", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def convert_row(row: Dict[str, str]) -> Dict[str, Any]:
    return {key: as_number(value) for key, value in row.items()}


def percentiles_sql(expr: str, suffix: str) -> str:
    points = [
        ("p01", "0.01"),
        ("p05", "0.05"),
        ("p10", "0.10"),
        ("p25", "0.25"),
        ("p50", "0.50"),
        ("p75", "0.75"),
        ("p90", "0.90"),
        ("p95", "0.95"),
        ("p99", "0.99"),
    ]
    return ",\n       ".join(
        f"(percentile_cont({point}) within group (order by {expr}))::float8 as {name}_{suffix}"
        for name, point in points
    )


def pair_source_sql(table: str, sample_pairs: int, seed: float, exact: bool) -> str:
    cfg = TABLES[table]
    id_col = cfg["id"]
    emb_col = cfg["embedding"]
    if exact:
        return f"""
            select
                a.{id_col} as id_a,
                b.{id_col} as id_b,
                (a.{emb_col} <=> b.{emb_col})::float8 as dist
            from {table} a
            join {table} b on a.{id_col} < b.{id_col}
            where a.{emb_col} is not null
              and b.{emb_col} is not null
        """

    return f"""
        with seed as (select setseed({seed})),
        rows as (
            select
                {id_col} as id,
                row_number() over (order by {id_col}) as rn
            from {table}
            where {emb_col} is not null
        ),
        n as (
            select count(*)::int as cnt from rows
        ),
        draws as (
            select
                floor(random() * n.cnt + 1)::int as rn_a,
                floor(random() * n.cnt + 1)::int as rn_b
            from seed
            cross join n
            cross join generate_series(1, {sample_pairs})
            where n.cnt > 1
        ),
        sample as (
            select r1.id as id_a, r2.id as id_b
            from draws d
            join rows r1 on r1.rn = d.rn_a
            join rows r2 on r2.rn = d.rn_b
            where r1.id <> r2.id
        )
        select
            s.id_a,
            s.id_b,
            (a.{emb_col} <=> b.{emb_col})::float8 as dist
        from sample s
        join {table} a on a.{id_col} = s.id_a
        join {table} b on b.{id_col} = s.id_b
    """


def table_overview(runner: PsqlRunner, table: str) -> Dict[str, Any]:
    emb_col = TABLES[table]["embedding"]
    sql = f"""
        select
            count(*)::bigint as rows,
            count({emb_col})::bigint as non_null_embeddings,
            count(*) filter (where {emb_col} is null)::bigint as null_embeddings,
            min(vector_dims({emb_col}))::int as min_dims,
            max(vector_dims({emb_col}))::int as max_dims,
            min(vector_norm({emb_col}))::float8 as min_norm,
            avg(vector_norm({emb_col}))::float8 as avg_norm,
            max(vector_norm({emb_col}))::float8 as max_norm
        from {table}
    """
    rows = runner.query_csv(sql)
    return convert_row(rows[0])


def pair_stats(
    runner: PsqlRunner,
    table: str,
    sample_pairs: int,
    exact_threshold: int,
    seed: float,
) -> Dict[str, Any]:
    overview = table_overview(runner, table)
    non_null = int(overview["non_null_embeddings"] or 0)
    exact = non_null <= exact_threshold
    source = pair_source_sql(table, sample_pairs, seed, exact)
    sql = f"""
        with pairs as ({source})
        select
            count(*)::bigint as pairs,
            min(dist)::float8 as min_distance,
            {percentiles_sql("dist", "distance")},
            avg(dist)::float8 as avg_distance,
            max(dist)::float8 as max_distance,
            min(1.0 - dist)::float8 as min_similarity,
            {percentiles_sql("1.0 - dist", "similarity")},
            avg(1.0 - dist)::float8 as avg_similarity,
            max(1.0 - dist)::float8 as max_similarity
        from pairs
    """
    rows = runner.query_csv(sql)
    stats = convert_row(rows[0])
    stats["mode"] = "exact" if exact else "sample"
    stats["requested_sample_pairs"] = None if exact else sample_pairs
    return {"overview": overview, "pairwise": stats}


def pair_examples(
    runner: PsqlRunner,
    table: str,
    sample_pairs: int,
    exact_threshold: int,
    seed: float,
    top_k: int,
    nearest: bool,
) -> List[Dict[str, Any]]:
    overview = table_overview(runner, table)
    non_null = int(overview["non_null_embeddings"] or 0)
    exact = non_null <= exact_threshold
    source = pair_source_sql(table, sample_pairs, seed, exact)
    cfg = TABLES[table]
    label_a = ", ".join(f"a.{col} as {col}_a" for col in cfg["label_columns"])
    label_b = ", ".join(f"b.{col} as {col}_b" for col in cfg["label_columns"])
    order = "asc" if nearest else "desc"
    comma_a = f", {label_a}" if label_a else ""
    comma_b = f", {label_b}" if label_b else ""
    sql = f"""
        with pairs as ({source})
        select
            p.id_a,
            p.id_b,
            p.dist as cosine_distance,
            (1.0 - p.dist)::float8 as cosine_similarity
            {comma_a}
            {comma_b}
        from pairs p
        join {table} a on a.{cfg["id"]} = p.id_a
        join {table} b on b.{cfg["id"]} = p.id_b
        order by p.dist {order}
        limit {top_k}
    """
    return [convert_row(row) for row in runner.query_csv(sql)]


def matched_image_text_stats(runner: PsqlRunner) -> Dict[str, Any]:
    sql = """
        with pairs as (
            select
                i.id as image_id,
                t.id as text_id,
                (i.embedding <=> t.embedding)::float8 as dist
            from image_chunks i
            join text_chunks t on t.id = i.text_chunk_id
            where i.embedding is not null
              and t.embedding is not null
        )
        select
            count(*)::bigint as pairs,
            min(dist)::float8 as min_distance,
            (percentile_cont(0.05) within group (order by dist))::float8 as p05_distance,
            (percentile_cont(0.50) within group (order by dist))::float8 as p50_distance,
            (percentile_cont(0.95) within group (order by dist))::float8 as p95_distance,
            avg(dist)::float8 as avg_distance,
            max(dist)::float8 as max_distance,
            min(1.0 - dist)::float8 as min_similarity,
            (percentile_cont(0.05) within group (order by 1.0 - dist))::float8 as p05_similarity,
            (percentile_cont(0.50) within group (order by 1.0 - dist))::float8 as p50_similarity,
            (percentile_cont(0.95) within group (order by 1.0 - dist))::float8 as p95_similarity,
            avg(1.0 - dist)::float8 as avg_similarity,
            max(1.0 - dist)::float8 as max_similarity
        from pairs
    """
    row = convert_row(runner.query_csv(sql)[0])
    return row


def image_same_path_stats(runner: PsqlRunner, sample_pairs: int, seed: float) -> List[Dict[str, Any]]:
    sql = f"""
        with seed as (select setseed({seed})),
        rows as (
            select
                id,
                image_path,
                embedding,
                row_number() over(order by id) as rn
            from image_chunks
            where embedding is not null
        ),
        n as (
            select count(*)::int as cnt from rows
        ),
        draws as (
            select
                floor(random() * n.cnt + 1)::int as rn_a,
                floor(random() * n.cnt + 1)::int as rn_b
            from seed
            cross join n
            cross join generate_series(1, {sample_pairs})
            where n.cnt > 1
        ),
        pairs as (
            select
                (a.image_path = b.image_path) as same_image,
                (a.embedding <=> b.embedding)::float8 as dist
            from draws d
            join rows a on a.rn = d.rn_a
            join rows b on b.rn = d.rn_b
            where a.id <> b.id
        )
        select
            same_image,
            count(*)::bigint as pairs,
            min(dist)::float8 as min_distance,
            (percentile_cont(0.50) within group (order by dist))::float8 as p50_distance,
            (percentile_cont(0.95) within group (order by dist))::float8 as p95_distance,
            avg(dist)::float8 as avg_distance,
            max(dist)::float8 as max_distance,
            min(1.0 - dist)::float8 as min_similarity,
            (percentile_cont(0.50) within group (order by 1.0 - dist))::float8 as p50_similarity,
            (percentile_cont(0.95) within group (order by 1.0 - dist))::float8 as p95_similarity,
            avg(1.0 - dist)::float8 as avg_similarity,
            max(1.0 - dist)::float8 as max_similarity
        from pairs
        group by same_image
        order by same_image desc
    """
    return [convert_row(row) for row in runner.query_csv(sql)]


def duplicate_summary(runner: PsqlRunner, table: str) -> Dict[str, Any]:
    if table == "image_chunks":
        summary_sql = """
            with groups as (
                select image_path, bbox, count(*)::bigint as n
                from image_chunks
                group by image_path, bbox
            )
            select
                (select count(*)::bigint from image_chunks) as rows,
                (select count(distinct (image_path, bbox))::bigint from image_chunks) as distinct_image_bbox,
                (select count(distinct embedding::text)::bigint from image_chunks) as distinct_embeddings,
                count(*) filter (where n > 1)::bigint as duplicate_groups,
                coalesce(sum(n) filter (where n > 1), 0)::bigint as rows_in_duplicate_groups,
                coalesce(sum(n - 1) filter (where n > 1), 0)::bigint as extra_duplicate_rows,
                max(n)::bigint as max_group_size
            from groups
        """
        top_sql = """
            select
                image_path,
                bbox,
                count(*)::bigint as rows,
                array_agg(id order by id)::text as ids,
                array_agg(text_chunk_id order by id)::text as text_chunk_ids
            from image_chunks
            group by image_path, bbox
            having count(*) > 1
            order by count(*) desc, image_path, bbox
            limit 10
        """
    elif table == "text_chunks":
        summary_sql = """
            with groups as (
                select content, count(*)::bigint as n
                from text_chunks
                group by content
            )
            select
                (select count(*)::bigint from text_chunks) as rows,
                (select count(distinct content)::bigint from text_chunks) as distinct_content,
                (select count(distinct embedding::text)::bigint from text_chunks) as distinct_embeddings,
                count(*) filter (where n > 1)::bigint as duplicate_groups,
                coalesce(sum(n) filter (where n > 1), 0)::bigint as rows_in_duplicate_groups,
                coalesce(sum(n - 1) filter (where n > 1), 0)::bigint as extra_duplicate_rows,
                max(n)::bigint as max_group_size
            from groups
        """
        top_sql = """
            select
                left(content, 160) as content_prefix,
                count(*)::bigint as rows,
                array_agg(id order by id)::text as ids
            from text_chunks
            group by content
            having count(*) > 1
            order by count(*) desc, left(content, 160)
            limit 10
        """
    else:
        raise ValueError(f"Unsupported table for duplicate summary: {table}")

    return {
        "summary": convert_row(runner.query_csv(summary_sql)[0]),
        "top_groups": [convert_row(row) for row in runner.query_csv(top_sql)],
    }


def collapse_warnings(stats: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    pairwise = stats["pairwise"]
    avg_sim = pairwise.get("avg_similarity")
    p50_sim = pairwise.get("p50_similarity")
    p95_sim = pairwise.get("p95_similarity")
    if avg_sim is not None and avg_sim > 0.95:
        warnings.append(f"avg similarity is very high ({avg_sim:.6f})")
    if p50_sim is not None and p50_sim > 0.98:
        warnings.append(f"median similarity is near-collapse ({p50_sim:.6f})")
    if p95_sim is not None and p95_sim > 0.995:
        warnings.append(f"p95 similarity is almost identical ({p95_sim:.6f})")
    return warnings


def print_table_report(table: str, report: Dict[str, Any]) -> None:
    overview = report["stats"]["overview"]
    pairwise = report["stats"]["pairwise"]
    print(f"\n== {table} ==")
    print(
        "rows={rows} non_null={non_null_embeddings} dims={min_dims}..{max_dims} "
        "norm(avg/min/max)={avg_norm:.6f}/{min_norm:.6f}/{max_norm:.6f}".format(**overview)
    )
    print(
        "pairwise mode={mode} pairs={pairs} "
        "distance avg/p50/p95/min/max={avg_distance:.6f}/{p50_distance:.6f}/"
        "{p95_distance:.6f}/{min_distance:.6f}/{max_distance:.6f}".format(**pairwise)
    )
    print(
        "similarity avg/p50/p95/min/max={avg_similarity:.6f}/{p50_similarity:.6f}/"
        "{p95_similarity:.6f}/{min_similarity:.6f}/{max_similarity:.6f}".format(**pairwise)
    )
    warnings = report["warnings"]
    if warnings:
        print("warnings: " + "; ".join(warnings))
    else:
        print("warnings: none")
    dup = report.get("duplicates", {}).get("summary")
    if dup:
        duplicate_bits = [
            f"duplicate_groups={dup.get('duplicate_groups')}",
            f"extra_duplicate_rows={dup.get('extra_duplicate_rows')}",
            f"max_group_size={dup.get('max_group_size')}",
        ]
        if "distinct_image_bbox" in dup:
            duplicate_bits.insert(0, f"distinct_image_bbox={dup.get('distinct_image_bbox')}")
        if "distinct_content" in dup:
            duplicate_bits.insert(0, f"distinct_content={dup.get('distinct_content')}")
        duplicate_bits.append(f"distinct_embeddings={dup.get('distinct_embeddings')}")
        print("duplicates: " + " ".join(duplicate_bits))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pgvector cosine distances in vectordb.")
    parser.add_argument("--jdbc", default="jdbc:postgresql://localhost:5432/vectordb")
    parser.add_argument("--user", default="user")
    parser.add_argument("--password", default="password")
    parser.add_argument("--docker-container", default="fffaffafaf-pgvector-1")
    parser.add_argument("--sample-pairs", type=int, default=100_000)
    parser.add_argument("--exact-threshold", type=int, default=2_000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=float, default=0.42)
    parser.add_argument("--output", default="debug/db_vector_distance_report.json")
    args = parser.parse_args()

    container = args.docker_container or autodetect_container(parse_jdbc_url(args.jdbc).port)
    if container:
        print(f"Using Docker container: {container}")
    elif shutil.which("psql") is None:
        raise RuntimeError("Neither a Docker Postgres container nor host psql was found.")

    runner = PsqlRunner(
        jdbc=args.jdbc,
        user=args.user,
        password=args.password,
        docker_container=container,
    )

    report: Dict[str, Any] = {
        "jdbc": args.jdbc,
        "docker_container": container,
        "sample_pairs": args.sample_pairs,
        "exact_threshold": args.exact_threshold,
        "tables": {},
    }

    for table in TABLES:
        stats = pair_stats(runner, table, args.sample_pairs, args.exact_threshold, args.seed)
        table_report = {
            "stats": stats,
            "warnings": collapse_warnings(stats),
            "duplicates": duplicate_summary(runner, table),
            "nearest_pairs": pair_examples(
                runner, table, args.sample_pairs, args.exact_threshold, args.seed, args.top_k, True
            ),
            "farthest_pairs": pair_examples(
                runner, table, args.sample_pairs, args.exact_threshold, args.seed, args.top_k, False
            ),
        }
        report["tables"][table] = table_report
        print_table_report(table, table_report)

    report["image_same_path_pairwise"] = image_same_path_stats(runner, args.sample_pairs, args.seed)
    print("\n== image_chunks same image_path split ==")
    for row in report["image_same_path_pairwise"]:
        label = "same image_path" if row["same_image"] else "different image_path"
        print(
            f"{label}: pairs={row['pairs']} "
            f"similarity avg/p50/p95/max={row['avg_similarity']:.6f}/"
            f"{row['p50_similarity']:.6f}/{row['p95_similarity']:.6f}/"
            f"{row['max_similarity']:.6f}"
        )

    report["matched_image_text"] = matched_image_text_stats(runner)
    mit = report["matched_image_text"]
    print("\n== matched image_chunks -> text_chunks ==")
    print(
        "pairs={pairs} distance avg/p50/p95/min/max={avg_distance:.6f}/{p50_distance:.6f}/"
        "{p95_distance:.6f}/{min_distance:.6f}/{max_distance:.6f}".format(**mit)
    )
    print("similarity avg/p50/p95={avg_similarity:.6f}/{p50_similarity:.6f}/{p95_similarity:.6f}".format(**mit))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved report: {args.output}")


if __name__ == "__main__":
    main()
