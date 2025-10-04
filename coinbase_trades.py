#!/usr/bin/env python3
import requests, time, sys, argparse, datetime as dt, csv
from datetime import timedelta

BASE = "https://api.exchange.coinbase.com"

def iso_to_epoch_ms(s):
    return int(dt.datetime.fromisoformat(s.replace("Z","+00:00")).timestamp() * 1000)

def fetch_trades(product_id="USDT-USDC", start_iso=None, end_iso=None, limit=1000, max_pages=None, csv_file=None):
    """
    Streams historical trades (most recent -> older) using Coinbase Exchange REST pagination.
    - start_iso: earliest ISO8601 time to include (older boundary)
    - end_iso: latest ISO8601 time to include (newer boundary)
    - csv_file: if provided, writes trades to this CSV file
    """
    params = {"limit": min(limit, 1000)}
    collected = 0
    pages = 0

    # Convert bounds to epoch ms for quick compare
    start_ms = iso_to_epoch_ms(start_iso) if start_iso else None
    end_ms   = iso_to_epoch_ms(end_iso)   if end_iso   else None

    session = requests.Session()
    url = f"{BASE}/products/{product_id}/trades"

    while True:
        r = session.get(url, params=params, timeout=20)
        r.raise_for_status()
        trades = r.json()

        # Coinbase returns newest-first. Filter by time window if provided.
        for t in trades:
            t_ms = iso_to_epoch_ms(t["time"])
            if end_ms and t_ms > end_ms:
                # too new; skip but keep paginating older
                continue
            if start_ms and t_ms < start_ms:
                # we've gone past the older bound; stop entirely
                return
            if csv_file:
                # Write to CSV if file is provided
                if collected == 0:  # Write header for first row
                    writer = csv.DictWriter(csv_file, fieldnames=t.keys())
                    writer.writeheader()
                writer = csv.DictWriter(csv_file, fieldnames=t.keys())
                writer.writerow(t)
            else:
                print(t)  # write each trade as a JSON-like dict line to stdout
            collected += 1

        pages += 1
        if max_pages and pages >= max_pages:
            return

        # Go older using the CB-AFTER cursor
        after = r.headers.get("CB-AFTER")
        if not after or len(trades) == 0:
            return
        params["after"] = after  # “after” means older page per docs

def main():
    ap = argparse.ArgumentParser(description="Download Coinbase historical trades for a product (USDT-USDC).")
    ap.add_argument("--product", default="USDT-USDC")
    ap.add_argument("--start", help="Earliest ISO8601 time to include (e.g. 2025-09-01T00:00:00Z)")
    ap.add_argument("--end", help="Latest ISO8601 time to include (e.g. 2025-10-01T00:00:00Z)")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--max-pages", type=int, help="Stop after this many pages (safety cap).")
    ap.add_argument("--last-10-days", action="store_true", help="Fetch trades from the last 10 days")
    ap.add_argument("--output-csv", help="Save trades to this CSV file")
    args = ap.parse_args()

    # Handle last 10 days option
    if args.last_10_days:
        end_date = dt.datetime.utcnow()
        start_date = end_date - timedelta(days=10)
        args.start = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        args.end = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        # Open CSV file if specified
        csv_file = open(args.output_csv, 'w', newline='') if args.output_csv else None
        try:
            fetch_trades(product_id=args.product, start_iso=args.start, end_iso=args.end,
                        limit=args.limit, max_pages=args.max_pages, csv_file=csv_file)
        finally:
            if csv_file:
                csv_file.close()
    except requests.HTTPError as e:
        sys.stderr.write(f"HTTP error: {e} | body={e.response.text}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
