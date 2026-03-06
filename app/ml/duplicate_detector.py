import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def detect_duplicates(transactions: list[dict]) -> dict[str, dict]:
    """Detect duplicate transactions based on same vendor, amount, and check_date.

    Returns a mapping of voucher_number -> {"is_duplicate": bool, "duplicate_of": str | None}.
    The first occurrence in each group is kept as the original; subsequent ones are flagged.
    """
    groups: dict[tuple, list[str]] = defaultdict(list)

    for txn in transactions:
        key = (
            (txn.get("vendor_name") or txn.get("description") or "").strip().lower(),
            str(txn["amount"]),
            str(txn["check_date"]),
        )
        groups[key].append(txn["voucher_number"])

    results: dict[str, dict] = {}

    for _key, vouchers in groups.items():
        # First occurrence is the original
        results[vouchers[0]] = {"is_duplicate": False, "duplicate_of": None}
        for dup in vouchers[1:]:
            results[dup] = {"is_duplicate": True, "duplicate_of": vouchers[0]}

    duplicate_count = sum(1 for r in results.values() if r["is_duplicate"])
    logger.info("Duplicate detection complete — %d duplicates found in %d transactions",
                duplicate_count, len(transactions))

    return results
