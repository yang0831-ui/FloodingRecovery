# -*- coding: utf-8 -*-

import os
import re
import io
import arcpy

# ===================== set directory =====================
COUNTY_SHP = r"E:\National University of Singapore\Yang Yang - flooding\Raw Data\SHP\selectedCounty.shp"
TILE_SHP   = r"E:\National University of Singapore\Yang Yang - flooding\Raw Data\SHP\BlackMarbleTiles.shp"

PROCESS_ROOT = r"E:\ProcessData"
OUT_MULTI_TILE_TXT = os.path.join(PROCESS_ROOT, "multi_tile_ctfips.txt")
OUT_DELETED_TXT    = os.path.join(PROCESS_ROOT, "deleted_items.txt")  

COUNTY_CODE_FIELD = "CTFIPS"   # county id


# ===================== Utility function =====================

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_tile_geoms(tile_shp):
    geoms = []
    desc = arcpy.Describe(tile_shp)
    sr   = desc.spatialReference
    with arcpy.da.SearchCursor(tile_shp, ["SHAPE@"]) as cur:
        for (geom,) in cur:
            if geom:
                geoms.append(geom)
    return geoms, sr

def get_multi_tile_ctfips(county_shp, tile_shp, county_code_field="CTFIPS"):
    multi = set()

    # read Tile Geometry and SR
    tile_geoms, sr_tile = load_tile_geoms(tile_shp)

    # County layer SR
    sr_county = arcpy.Describe(county_shp).spatialReference

    # Iterate through each county
    fields = [county_code_field, "SHAPE@"]
    with arcpy.da.SearchCursor(county_shp, fields) as cur:
        for ctfips, cgeom in cur:
            if cgeom is None:
                continue

            if sr_tile and sr_tile.name and sr_county.name != sr_tile.name:
                try:
                    cgeom_use = cgeom.projectAs(sr_tile)
                except Exception:
                    cgeom_use = cgeom 
            else:
                cgeom_use = cgeom

            cnt = 0
            for tgeom in tile_geoms:
                # quickly skip disjoint geometries
                try:
                    if cgeom_use.disjoint(tgeom):
                        continue
                except Exception:
                    pass
                try:
                    inter = cgeom_use.intersect(tgeom, 4)  # 4 = esriGeometryPolygon
                    if inter and inter.area > 0:
                        cnt += 1
                        if cnt >= 2:
                            multi.add(str(ctfips))
                            break
                except Exception:
                    continue

    return multi

def list_processdata_dirs(root):
    out = []
    if not os.path.isdir(root):
        return out
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and name.startswith("ProcessData_"):
            # State code to be retrieved
            parts = name.split("_", 1)
            state = parts[1] if len(parts) == 2 else name.replace("ProcessData_", "")
            state = state.strip()
            if state:
                out.append((p, state))
    return out

def delete_file_silent(path):
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
    except Exception:
        pass
    return False

def remove_ctfips_from_progress(progress_txt, ctfips):
    if not os.path.exists(progress_txt):
        return False

    try:
        with io.open(progress_txt, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # if UTF-8-SIG fails, try UTF-8
        with open(progress_txt, "r", encoding="utf-8") as f:
            lines = f.readlines()

    pattern = re.compile(rf"\b{re.escape(str(ctfips))}\b")
    new_lines = [ln for ln in lines if not pattern.search(ln)]

    if len(new_lines) != len(lines):
        with io.open(progress_txt, "w", encoding="utf-8-sig", newline="") as f:
            f.writelines(new_lines)
        return True
    return False

# ===================== Main Process =====================

def main():
    # 1) calculate multi-tile counties
    print("[1/3] checking multi-tile counties...")
    multi_ctfips = get_multi_tile_ctfips(COUNTY_SHP, TILE_SHP, COUNTY_CODE_FIELD)
    print(f"   ✓ Found multi-tile counties: {len(multi_ctfips)}")

    # Write out multi_tile_ctfips.txt
    _ensure_dir(OUT_MULTI_TILE_TXT)
    with io.open(OUT_MULTI_TILE_TXT, "w", encoding="utf-8-sig", newline="") as f:
        for code in sorted(multi_ctfips):
            f.write(f"{code}\n")
    print(f"   ✓ Already done: {OUT_MULTI_TILE_TXT}")

    # 2) Iterate through ProcessData_XX and perform deletion
    print("[2/3] Iterating through E:\\ProcessData for ProcessData_XX ...")
    pd_dirs = list_processdata_dirs(PROCESS_ROOT)
    print(f"   Found {len(pd_dirs)} subdirectories.")

    deleted_pairs = set()  

    for pd_path, state in pd_dirs:
        lighting_dir = os.path.join(pd_path, "lighting_csv")
        progress_txt = os.path.join(pd_path, "progress.txt")

        for ctfips in multi_ctfips:
            removed_any = False

            # Delete CSV
            csv_path = os.path.join(lighting_dir, f"{ctfips}.csv")
            if delete_file_silent(csv_path):
                removed_any = True
                print(f"   - Deleted file: {csv_path}")

            # Remove record from progress.txt
            if remove_ctfips_from_progress(progress_txt, ctfips):
                removed_any = True
                print(f"   - Updated progress: {progress_txt} (removed {ctfips})")

            if removed_any:
                deleted_pairs.add((state, ctfips))

    # 3) Write out deletion list
    print("[3/3] Writing out deletion list...")
    _ensure_dir(OUT_DELETED_TXT)
    with io.open(OUT_DELETED_TXT, "w", encoding="utf-8-sig", newline="") as f:
        # Sort by state code and county code
        for state, ctfips in sorted(deleted_pairs, key=lambda x: (x[0], x[1])):
            f.write(f"{state}{ctfips}\n")
    print(f"   ✓ Already done: {OUT_DELETED_TXT}")
    print("Done.")

if __name__ == "__main__":
    main()
