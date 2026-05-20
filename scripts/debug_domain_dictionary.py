# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse

from rag.domain_1c_dictionary import (
    query_to_targets,
    repair_ui_item_text,
    canonicalize_1c_term,
    normalize_1c_text,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    print("QUERY:", args.query)
    print("NORMALIZED:", normalize_1c_text(args.query))
    print("CANONICAL:", canonicalize_1c_term(args.query))
    print("REPAIRED:", repair_ui_item_text(args.query))
    print("TARGETS:")
    for t in query_to_targets(args.query):
        print(" -", t)


if __name__ == "__main__":
    main()