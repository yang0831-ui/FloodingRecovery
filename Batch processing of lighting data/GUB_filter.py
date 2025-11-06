# arcpy
import arcpy, glob, logging, csv
from pathlib import Path
import pandas as pd

# Set paths
ROOT_IN   = Path(r"E:\ProcessData")
ROOT_OUT  = Path(r"E:\ProcessData_selected")
GUB_FC    = Path(r"E:\National University of Singapore\Yang Yang - flooding\Raw Data\SHP\建成区\GUB_Selected_2020.shp")

arcpy.env.overwriteOutput = True
arcpy.env.parallelProcessingFactor = "75%"

# logging setup
logging.basicConfig(
    filename="gub_filter.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# used for resuming
DONE_CSVS = {p.name for p in ROOT_OUT.rglob("lighting_csv/*.csv")}


gub_lyr = "gub_lyr"
arcpy.management.MakeFeatureLayer(str(GUB_FC), gub_lyr)

def grid_xyids(grid_fc: str) -> list[str]:
    lyr = "grid_lyr"
    arcpy.management.MakeFeatureLayer(grid_fc, lyr)
    arcpy.management.SelectLayerByLocation(
        in_layer=lyr,
        overlap_type="INTERSECT",
        select_features=gub_lyr,
        selection_type="NEW_SELECTION"
    )
    cnt = int(arcpy.management.GetCount(lyr)[0])
    if cnt == 0:
        return []
    arr = arcpy.da.FeatureClassToNumPyArray(lyr, ["xy_id"])
    return arr["xy_id"].astype(str).tolist()

def filter_csv_in_chunks(csv_path: Path, keep_ids: set[str], out_csv: Path, chunksize: int = 100_000) -> int:
    total_written = 0
    if out_csv.exists():
        out_csv.unlink()
    reader = pd.read_csv(csv_path, dtype=str, low_memory=False, chunksize=chunksize, on_bad_lines="skip")
    for chunk in reader:
        if "xy_id" not in chunk.columns:
            chunk = chunk.rename_axis("xy_id").reset_index()
        filtered = chunk[chunk["xy_id"].isin(keep_ids)]
        if filtered.empty:
            continue
        filtered.to_csv(
            out_csv,
            index=False,
            mode="w" if total_written == 0 else "a",
            header=(total_written == 0),
        )
        total_written += len(filtered)
    if total_written == 0 and out_csv.exists():
        out_csv.unlink()
    return total_written

def filter_csv_stream(csv_path: Path, keep_ids: set[str], out_csv: Path,
                      input_encoding: str = "utf-8", 
                      output_encoding: str = "utf-8") -> int:
    total = 0
    if out_csv.exists():
        out_csv.unlink()

    with open(csv_path, "r", encoding=input_encoding, newline="") as rf, \
         open(out_csv, "w", encoding=output_encoding, newline="") as wf:

        reader = csv.reader(rf)
        writer = csv.writer(wf)

        # Write header
        try:
            header = next(reader)
        except StopIteration:
            return 0
        writer.writerow(header)

        # find xy_id column
        try:
            idx = header.index("xy_id")
        except ValueError:
            idx = 0

        # filter by rows
        for row in reader:
            if idx < len(row) and row[idx] in keep_ids:
                writer.writerow(row)
                total += 1

    if total == 0 and out_csv.exists():
        out_csv.unlink()
    return total

def process_state(state_dir: Path):
    # Process a single state
    fishnet_dir  = state_dir / "fishnet"
    lighting_dir = state_dir / "lighting_csv"
    out_dir      = ROOT_OUT / state_dir.name / "lighting_csv"

    if not (fishnet_dir.exists() and lighting_dir.exists()):
        print(f"[Skip] {state_dir.name} (missing fishnet or lighting_csv)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for shp in fishnet_dir.glob("*.shp"):
        county = shp.stem.split("_")[0]
        csv_list = list(lighting_dir.glob(f"{county}*.csv"))
        if not csv_list:
            print(f"  ⚠ {county}: No matching CSV found, skipping")
            continue

        csv_path = csv_list[0]
        out_csv  = out_dir / csv_path.name

        # used for resuming
        if out_csv.name in DONE_CSVS and out_csv.exists():
            print(f"  ↪ {county}: Already done, skipping")
            continue

        # Calculate keep xy_id
        keep = set(grid_xyids(str(shp)))
        if not keep:
            print(f"  {county}: No intersecting cells, skip them.")
            continue

        # Check file size
        size_bytes = csv_path.stat().st_size
        size_gb = size_bytes / (1024**3)

        print(f"  ▶ {county}: File size {size_gb:.2f} GB, using " +
              ("chunked Pandas filtering" if size_gb < 1 else "streaming filtering"))

        if size_gb < 1:
            # Small file: Pandas chunking
            written = filter_csv_in_chunks(csv_path, keep, out_csv, chunksize=100_000)
        else:
            # Large file: Streaming
            written = filter_csv_stream(csv_path, keep, out_csv,
                                        input_encoding="utf-8",
                                        output_encoding="utf-8")

        if written:
            logging.info(f"Wrote {written} rows → {out_csv}")
            print(f"      ✓ Already written {written} rows")
        else:
            print(f"      ⚠ {county}: No matching rows, file not generated")


def main():
    print("=== Start Processing ===")
    for sd in ROOT_IN.iterdir():
        if sd.is_dir() and sd.name.startswith("ProcessData_"):
            print(f"\n>> {sd.name}")
            process_state(sd)
    print("\n=== All Done ===")

if __name__ == "__main__":
    main()
