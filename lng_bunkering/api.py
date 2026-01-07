#!/usr/bin/env python
# coding: utf-8

import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.interpolate import RegularGridInterpolator
from config import DATA_DIR

# FastAPI app
app = FastAPI(
    title="LNG Bunkering Application API",
    description="REST API for LNG Bunkering Operations (no persistence)",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
# base_dir = os.path.dirname(os.path.abspath(__file__))
ship_dir = DATA_DIR

# Available ships configuration
available_ships: Dict[str, List[str]] = {
    "MOUNT TOURMALINE": ["LNG_TK1", "LNG_TK2"],
    "MOUNT NOVATERRA": ["LNG_TK1", "LNG_TK2"],
    "MOUNT ANETO": ["LNG_TK1", "LNG_TK2"],
    "MOUNT TAI": ["LNG_TK1", "LNG_TK2"],
    "MOUNT OSSA": ["LNG_TK1", "LNG_TK2"],
    "MOUNT JADEITE": ["LNG_TK1", "LNG_TK2"],
    "MOUNT API": ["LNG_TK1", "LNG_TK2"],
    "MOUNT AMELIOR": ["LNG_TK1", "LNG_TK2"],
    "MOUNT HENG": ["LNG_TK1", "LNG_TK2"],
    "MOUNT GOWER": ["LNG_TK1", "LNG_TK2"],
    "MOUNT GAEA": ["LNG_TK1", "LNG_TK2"],
    "MOUNT COOK": ["LNG_TK1", "LNG_TK2"],
    "MOUNT ARARAT": ["LNG_TK1", "LNG_TK2"],
    "ATLANTIC PEARL": ["LNG_TK1", "LNG_TK2"],
    "CMA CGM ARCTIC": ["LNG_TK"],
    "CMA CGM BALI": ["LNG_TK"],
    "CMA CGM DIGNITY": ["LNG_TK"],
    "CMA CGM HOPE": ["LNG_TK"],
    "CMA CGM IGUACU": ["LNG_TK"],
    "CMA CGM INTEGRITY": ["LNG_TK"],
    "CMA CGM LIBERTY": ["LNG_TK"],
    "CMA CGM TENERE": ["LNG_TK"],
    "CMA CGM PRIDE": ["LNG_TK"],
    "CMA CGM SCANDOLA": ["LNG_TK"],
    "CMA CGM SYMI": ["LNG_TK"],
    "CMA CGM UNITY": ["LNG_TK"],
    "ZIM ARIES": ["LNG_TANK"],
    "ZIM GEMINI": ["LNG_TANK"],
    "ZIM SCORPIO": ["LNG_TANK"],
    "QUETZAL": ["LNGAS_TK"],
    "COPAN": ["LNGAS_TK"],
    "TISCAPA": ["LNGAS_TK"],
    "CMA CGM DAYTONA": ["LNG_TK1", "LNG_TK2"],
    "CMA CGM INDIANAPOLIS": ["LNG_TK1", "LNG_TK2"],
    "CMA CGM MONACO": ["LNG_TK1", "LNG_TK2"],
    "CMA CGM SILVERSTONE": ["LNG_TK1", "LNG_TK2"],
    "CMA CGM MONZA": ["LNG_TK1", "LNG_TK2"],
    "LAKE HERMAN": ["LNG_TK1", "LNG_TK2"],
    "LAKE ANNECY": ["LNG_TK1", "LNG_TK2"],
    "LAKE LUGU": ["LNG_TK1", "LNG_TK2"],
    "LAKE QARAOUN": ["LNG_TK1", "LNG_TK2"],
    "LAKE SAINT ANNE": ["LNG_TK1", "LNG_TK2"],
    "LAKE TRAVIS": ["LNG_TK1", "LNG_TK2"],
    "LAKE TAZAWA": ["LNG_TK1", "LNG_TK2"],
    "ATLANTIC JADE": ["LNG_TK1", "LNG_TK2"],
    "ATLANTIC EMERALD": ["LNG_TK1", "LNG_TK2"],
    "STARWAY": ["LNG_TK1", "LNG_TK2"],
    "GREENWAY": ["LNG_TK1", "LNG_TK2"],
}

# Tank flags (by family)
lng_tanks = ["LNG_TK"]        # tanks needing only list/trim corrections
lng_tks = ["LNG_TANK"]        # ZIM family: list/trim + volume
LNG_TK_ALIAS = ["LNGAS_TK"]   # alias treated like lng_tanks

# Models
class TankInput(BaseModel):
    level: float = Field(..., description="Tank level in mm")
    vapor_temp: float = Field(..., description="Vapor temperature in °C")
    liquid_temp: float = Field(..., description="Liquid temperature in °C")
    pressure: float = Field(..., description="Gauge pressure in Bar")

class BunkeringRequest(BaseModel):
    ship_id: str = Field(..., description="Ship identifier")
    opening_tank1: TankInput
    opening_tank2: Optional[TankInput] = None
    closing_tank1: TankInput
    closing_tank2: Optional[TankInput] = None

    opening_trim: float = Field(..., description="Opening trim in m")
    opening_list: float = Field(..., description="Opening list in degrees")
    closing_trim: float = Field(..., description="Closing trim in m")
    closing_list: float = Field(..., description="Closing list in degrees")

    opening_time: str = Field(..., description="Opening time (MM/DD/YYYY HH:MM)")
    closing_time: str = Field(..., description="Closing time (MM/DD/YYYY HH:MM)")

    density: float = Field(..., description="Density in kg/m3")
    bdn_quantity: float = Field(..., description="BDN quantity in m3")
    bog: float = Field(..., description="Average BOG in kg/h")
    gross_energy: float = Field(..., description="Gross energy in MMBtu or MWh")
    unreckoned_qty: float = Field(..., description="Unreckoned quantity in m3")
    net_energy: float = Field(..., description="Net energy in MMBtu or MWh")

class BunkeringResponse(BaseModel):
    ship_id: str
    tank1_volume_opening: float
    tank2_volume_opening: Optional[float]
    tank1_volume_closing: float
    tank2_volume_closing: Optional[float]
    opening_quantity: float
    closing_quantity: float
    volume_difference: float
    bog_consumption: float
    loaded_quantity: float
    net_quantity: float
    difference: float
    calculation_time: str

# Utilities
def get_tank_data_path(ship_id: str, tank_id: str) -> Dict[str, str]:
    ship_data_dir = os.path.join(ship_dir, ship_id)
    if tank_id in lng_tanks or tank_id in LNG_TK_ALIAS:
        return {
            "list_table": os.path.join(ship_data_dir, f"list_table_{tank_id}.csv"),
            "trim_table": os.path.join(ship_data_dir, f"trim_table_{tank_id}.csv"),
        }
    elif tank_id in lng_tks:
        return {
            "list_table": os.path.join(ship_data_dir, f"list_table_{tank_id}.csv"),
            "trim_table": os.path.join(ship_data_dir, f"trim_table_{tank_id}.csv"),
            "volume_table": os.path.join(ship_data_dir, f"volume_table_{tank_id}.csv"),
        }
    else:
        return {
            "volume_table": os.path.join(ship_data_dir, f"volume_table_{tank_id}.csv"),
            "list_table": os.path.join(ship_data_dir, f"list_table_{tank_id}.csv"),
            "trim_table": os.path.join(ship_data_dir, f"trim_table_{tank_id}.csv"),
            "temp_table": os.path.join(ship_data_dir, f"temp_table_{tank_id}.csv"),
            "press_table": os.path.join(ship_data_dir, f"press_table_{tank_id}.csv"),
        }

def get_range_values(ship_id: str, tank_id: str):
    tank_paths = get_tank_data_path(ship_id, tank_id)
    if not all(os.path.exists(p) for p in tank_paths.values()):
        raise HTTPException(
            status_code=404,
            detail=f"Missing data files for ship {ship_id} and tank {tank_id}",
        )

    def extract_values(columns: List[str], prefix: str) -> List[float]:
        vals: List[float] = []
        for col in columns:
            m = re.search(rf"{prefix}([-+]?\d*\.?\d+)", col)
            if m:
                vals.append(float(m.group(1)))
        return vals

    if tank_id in lng_tanks or tank_id in LNG_TK_ALIAS:
        level_list_df = pd.read_csv(tank_paths["list_table"])
        level_trim_df = pd.read_csv(tank_paths["trim_table"])
        list_values = extract_values(list(level_list_df.columns[1:]), "list_")
        trim_values = extract_values(list(level_trim_df.columns[1:]), "trim_")
        level_min = float(level_trim_df["level"].min())
        level_max = float(level_trim_df["level"].max())
        list_min, list_max = (min(list_values), max(list_values)) if list_values else (None, None)
        trim_min, trim_max = (min(trim_values), max(trim_values)) if trim_values else (None, None)
        temp_min, temp_max = (-165.0 if tank_id in LNG_TK_ALIAS else -163.0), 20.0
        press_min, press_max = 0.0, 0.7
        return level_min, level_max, list_min, list_max, trim_min, trim_max, temp_min, temp_max, press_min, press_max

    elif tank_id in lng_tks:
        level_list_df = pd.read_csv(tank_paths["list_table"])
        level_trim_df = pd.read_csv(tank_paths["trim_table"])
        level_volume_df = pd.read_csv(tank_paths["volume_table"])
        list_values = extract_values(list(level_list_df.columns[1:]), "list_")
        trim_values = extract_values(list(level_trim_df.columns[1:]), "trim_")
        level_min = float(level_volume_df["level"].min())
        level_max = float(level_volume_df["level"].max())
        list_min, list_max = (min(list_values), max(list_values)) if list_values else (None, None)
        trim_min, trim_max = (min(trim_values), max(trim_values)) if trim_values else (None, None)
        temp_min, temp_max = -163.0, 20.0
        press_min, press_max = 0.0, 0.7
        return level_min, level_max, list_min, list_max, trim_min, trim_max, temp_min, temp_max, press_min, press_max

    else:
        required = ["volume_table", "list_table", "trim_table", "temp_table", "press_table"]
        if not all(k in tank_paths and os.path.exists(tank_paths[k]) for k in required):
            raise HTTPException(status_code=404, detail=f"Missing required files for {ship_id} {tank_id}")
        level_volume_df = pd.read_csv(tank_paths["volume_table"])
        level_list_df = pd.read_csv(tank_paths["list_table"])
        level_trim_df = pd.read_csv(tank_paths["trim_table"])
        level_temp_df = pd.read_csv(tank_paths["temp_table"])
        level_press_df = pd.read_csv(tank_paths["press_table"])

        def ex(columns: List[str], prefix: str) -> List[float]:
            return [float(col.replace(prefix, "")) for col in columns]

        list_values = ex(list(level_list_df.columns[1:]), "list_")
        trim_values = ex(list(level_trim_df.columns[1:]), "trim_")
        temp_values = ex(list(level_temp_df.columns[1:]), "temp_")
        press_values = ex(list(level_press_df.columns[1:]), "press_")

        level_min = float(level_volume_df["level"].min())
        level_max = float(level_volume_df["level"].max())
        list_min, list_max = (min(list_values), max(list_values)) if list_values else (None, None)
        trim_min, trim_max = (min(trim_values), max(trim_values)) if trim_values else (None, None)
        temp_min, temp_max = (min(temp_values), max(temp_values)) if temp_values else (None, None)
        press_min, press_max = (min(press_values), max(press_values)) if press_values else (None, None)
        return level_min, level_max, list_min, list_max, trim_min, trim_max, temp_min, temp_max, press_min, press_max

def compute_corrected_values(
    ship_id: str,
    tank_id: str,
    level: float,
    list_: float,
    trim_: float,
    temp_: float,
    press_: float,
):
    try:
        print(
            f"DEBUG: compute_corrected_values ship_id={ship_id}, tank_id={tank_id}, "
            f"level={level}, list_={list_}, trim_={trim_}"
        )
        tank_paths = get_tank_data_path(ship_id, tank_id)
        if not tank_paths:
            print(f"DEBUG: No tank paths found for {ship_id} - {tank_id}")
            return None, None

        if tank_id in lng_tanks or tank_id in LNG_TK_ALIAS:
            level_list_df = pd.read_csv(tank_paths["list_table"])
            level_trim_df = pd.read_csv(tank_paths["trim_table"])

            level_values_1 = level_list_df["level"].values
            level_values_2 = level_trim_df["level"].values
            list_values = np.array([float(c.replace("list_", "")) for c in level_list_df.columns[1:]])
            trim_values = np.array([float(c.replace("trim_", "")) for c in level_trim_df.columns[1:]])

            level_list_interpolator = RegularGridInterpolator(
                (level_values_1, list_values),
                level_list_df.iloc[:, 1:].values,
                method="linear",
            )
            level_trim_interpolator = RegularGridInterpolator(
                (level_values_2, trim_values),
                level_trim_df.iloc[:, 1:].values,
                method="linear",
            )

            list_correction = float(level_list_interpolator([[level, list_]])[0])
            corrected_level = level + list_correction
            corrected_volume = float(level_trim_interpolator([[corrected_level, trim_]])[0])

            print(f"DEBUG: computed corrected_level={corrected_level}, corrected_volume={corrected_volume}")
            return round(corrected_level, 2), round(corrected_volume, 2)

        elif tank_id in lng_tks:
            level_list_df = pd.read_csv(tank_paths["list_table"])
            level_trim_df = pd.read_csv(tank_paths["trim_table"])
            level_volume_df = pd.read_csv(tank_paths["volume_table"])

            level_values = level_volume_df["level"].values
            volume_values = level_volume_df["volume"].values
            list_values = np.array([float(c.replace("list_", "")) for c in level_list_df.columns[1:]])
            trim_values = np.array([float(c.replace("trim_", "")) for c in level_trim_df.columns[1:]])

            level_list_interpolator = RegularGridInterpolator(
                (level_values, list_values), level_list_df.iloc[:, 1:].values, method="linear"
            )
            level_trim_interpolator = RegularGridInterpolator(
                (level_values, trim_values), level_trim_df.iloc[:, 1:].values, method="linear"
            )
            level_volume_interpolator = RegularGridInterpolator(
                (level_values,), volume_values, method="linear"
            )

            list_correction = float(level_list_interpolator([[level, list_]])[0])
            trim_correction = float(level_trim_interpolator([[level, trim_]])[0])
            corrected_level = level + list_correction + trim_correction
            corrected_volume = float(level_volume_interpolator([[corrected_level]])[0])

            print(f"DEBUG: computed corrected_level={corrected_level}, corrected_volume={corrected_volume}")
            return round(corrected_level, 2), round(corrected_volume, 2)

        else:
            level_volume_df = pd.read_csv(tank_paths["volume_table"])
            level_list_df = pd.read_csv(tank_paths["list_table"])
            level_trim_df = pd.read_csv(tank_paths["trim_table"])
            level_temp_df = pd.read_csv(tank_paths["temp_table"])
            level_press_df = pd.read_csv(tank_paths["press_table"])

            level_values = level_volume_df["level"].values
            volume_values = level_volume_df["volume"].values
            list_values = np.array([float(c.replace("list_", "")) for c in level_list_df.columns[1:]])
            trim_values = np.array([float(c.replace("trim_", "")) for c in level_trim_df.columns[1:]])
            temp_values = np.array([float(c.replace("temp_", "")) for c in level_temp_df.columns[1:]])
            press_values = np.array([float(c.replace("press_", "")) for c in level_press_df.columns[1:]])

            level_list_interpolator = RegularGridInterpolator(
                (level_values, list_values), level_list_df.iloc[:, 1:].values, method="linear"
            )
            level_trim_interpolator = RegularGridInterpolator(
                (level_values, trim_values), level_trim_df.iloc[:, 1:].values, method="linear"
            )
            level_temp_interpolator = RegularGridInterpolator(
                (level_values, temp_values), level_temp_df.iloc[:, 1:].values, method="linear"
            )
            level_press_interpolator = RegularGridInterpolator(
                (level_values, press_values), level_press_df.iloc[:, 1:].values, method="linear"
            )
            level_volume_interpolator = RegularGridInterpolator(
                (level_values,), volume_values, method="linear"
            )

            list_correction = float(level_list_interpolator([[level, list_]])[0])
            trim_correction = float(level_trim_interpolator([[level, trim_]])[0])
            temp_correction = float(level_temp_interpolator([[level, temp_]])[0])
            press_correction = float(level_press_interpolator([[level, press_]])[0])

            corrected_level = level + list_correction + trim_correction + temp_correction + press_correction
            corrected_volume = float(level_volume_interpolator([[corrected_level]])[0])

            print(f"DEBUG: computed corrected_level={corrected_level}, corrected_volume={corrected_volume}")
            return round(corrected_level, 2), round(corrected_volume, 2)

    except Exception as e:
        print(f"DEBUG: Error in compute_corrected_values: {str(e)}")
        return None, None

def get_ship_parameters(ship_id: str) -> Dict[str, Any]:
    if ship_id in ["MOUNT TOURMALINE", "MOUNT NOVATERRA"]:
        return {"BOG_max": 500, "LNG_TK1_cap": 3175.139, "LNG_TK2_cap": 3180.121, "identity": "209k_bulk"}
    elif ship_id in [
        "MOUNT ANETO", "MOUNT TAI", "MOUNT OSSA", "MOUNT JADEITE", "MOUNT API",
        "MOUNT AMELIOR", "MOUNT HENG", "MOUNT GOWER", "MOUNT GAEA",
        "MOUNT COOK", "MOUNT ARARAT"
    ]:
        return {"BOG_max": 500, "LNG_TK1_cap": 3181.546, "LNG_TK2_cap": 3179.732, "identity": "210k_bulk"}
    elif ship_id in [
        "CMA CGM ARCTIC", "CMA CGM BALI", "CMA CGM DIGNITY", "CMA CGM HOPE",
        "CMA CGM IGUACU", "CMA CGM INTEGRITY", "CMA CGM LIBERTY", "CMA CGM PRIDE",
        "CMA CGM TENERE", "CMA CGM SCANDOLA", "CMA CGM SYMI", "CMA CGM UNITY"
    ]:
        return {"BOG_max": 500, "LNG_TK1_cap": 12448.3, "identity": "CMA_cont"}
    elif ship_id in ["ZIM ARIES", "ZIM GEMINI", "ZIM SCORPIO"]:
        return {"BOG_max": 1200, "LNG_TK1_cap": 6125.285, "identity": "ZIM_cont"}
    elif ship_id in [
        "CMA CGM DAYTONA", "CMA CGM INDIANAPOLIS", "CMA CGM MONACO", "CMA CGM SILVERSTONE",
        "CMA CGM MONZA", "LAKE HERMAN", "LAKE ANNECY", "LAKE LUGU", "LAKE QARAOUN",
        "LAKE SAINT ANNE", "LAKE TRAVIS", "LAKE TAZAWA"
    ]:
        return {"BOG_max": 600, "LNG_TK1_cap": 2013.699, "LNG_TK2_cap": 2014.748, "identity": "PCTC"}
    elif ship_id in ["ATLANTIC JADE", "ATLANTIC EMERALD"]:
        return {"BOG_max": 1200, "LNG_TK1_cap": 2324.113, "LNG_TK2_cap": 2322.097, "identity": "110k_tanker"}
    elif ship_id in ["ATLANTIC PEARL"]:   
            return {"BOG_max": 1200, "LNG_TK1_cap": 1816.435, "LNG_TK2_cap": 1818.006, "identity": "111k_tanker"}    
    elif ship_id in ["STARWAY", "GREENWAY"]:
        return {"BOG_max": 1200, "LNG_TK1_cap": 2570.133, "LNG_TK2_cap": 2571.517, "identity": "150k_tanker"}
    elif ship_id in ["QUETZAL", "COPAN", "TISCAPA"]:
        return {"BOG_max": 500, "LNG_TK1_cap": 1613, "identity": "1400TEU_cont"}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown ship ID: {ship_id}")

# Endpoints
@app.get("/")
async def root():
    return {"message": "LNG Bunkering Application API", "version": "1.0.0", "persistence": "none"}

@app.get("/ships")
async def get_ships():
    return {"ships": list(available_ships.keys())}

@app.get("/debug/files/{ship_id}")
async def debug_files(ship_id: str):
    if ship_id not in available_ships:
        raise HTTPException(status_code=404, detail="Ship not found")
    debug_info: Dict[str, Any] = {}
    for tank_id in available_ships[ship_id]:
        tank_paths = get_tank_data_path(ship_id, tank_id)
        debug_info[tank_id] = {
            "paths": tank_paths,
            "exists": {k: os.path.exists(p) for k, p in tank_paths.items()},
            "file_size": {k: (os.path.getsize(p) if os.path.exists(p) else 0) for k, p in tank_paths.items()},
        }
    return {"ship_id": ship_id, "ship_dir": ship_dir, "debug_info": debug_info}

@app.get("/ships/{ship_id}")
async def get_ship_details(ship_id: str):
    if ship_id not in available_ships:
        raise HTTPException(status_code=404, detail="Ship not found")
    tank_ids = available_ships[ship_id]
    file_status: Dict[str, Any] = {}
    for tank_id in tank_ids:
        tank_paths = get_tank_data_path(ship_id, tank_id)
        file_status[tank_id] = {
            "list_table": os.path.exists(tank_paths.get("list_table", "")),
            "trim_table": os.path.exists(tank_paths.get("trim_table", "")),
            "volume_table": os.path.exists(tank_paths.get("volume_table", "")) if "volume_table" in tank_paths else None,
            "temp_table": os.path.exists(tank_paths.get("temp_table", "")) if "temp_table" in tank_paths else None,
            "press_table": os.path.exists(tank_paths.get("press_table", "")) if "press_table" in tank_paths else None,
        }
    return {
        "ship_id": ship_id,
        "tanks": tank_ids,
        "tank_count": len(tank_ids),
        "file_status": file_status,
    }

@app.post("/bunkering/calculate", response_model=BunkeringResponse)
async def calculate_bunkering(
    request: BunkeringRequest = Body(
        examples={
            "TwoTankVessel": {
                "summary": "Two-tank vessel (e.g., CMA CGM MONACO)",
                "value": {
                    "ship_id": "CMA CGM MONACO",
                    "opening_tank1": {"level": 1000, "vapor_temp": -150, "liquid_temp": -160, "pressure": 0.22},
                    "opening_tank2": {"level": 980, "vapor_temp": -151, "liquid_temp": -158, "pressure": 0.20},
                    "closing_tank1": {"level": 1100, "vapor_temp": -149, "liquid_temp": -156, "pressure": 0.23},
                    "closing_tank2": {"level": 1085, "vapor_temp": -148, "liquid_temp": -155, "pressure": 0.24},
                    "opening_trim": 0.0, "opening_list": 0.0, "closing_trim": 0.0, "closing_list": 0.0,
                    "opening_time": "07/10/2025 10:00", "closing_time": "07/10/2025 16:00",
                    "density": 0.45, "bdn_quantity": 1000, "bog": 300,
                    "gross_energy": 10000, "unreckoned_qty": 0, "net_energy": 9800
                },
            },
            "OneTankVessel": {
                "summary": "One-tank vessel (e.g., CMA CGM ARCTIC)",
                "value": {
                    "ship_id": "CMA CGM ARCTIC",
                    "opening_tank1": {"level": 8500, "vapor_temp": -155, "liquid_temp": -160, "pressure": 0.18},
                    "closing_tank1": {"level": 8600, "vapor_temp": -154, "liquid_temp": -159, "pressure": 0.19},
                    "opening_trim": 0.0, "opening_list": 0.0, "closing_trim": 0.0, "closing_list": 0.0,
                    "opening_time": "07/10/2025 10:00", "closing_time": "07/10/2025 16:00",
                    "density": 0.45, "bdn_quantity": 1000, "bog": 300,
                    "gross_energy": 10000, "unreckoned_qty": 0, "net_energy": 9800
                },
            },
        }
    )
):
    try:
        print(f"=== DEBUG: Starting calculation for ship {request.ship_id} ===")
        if request.ship_id not in available_ships:
            raise HTTPException(status_code=400, detail="Invalid ship ID")

        tank_ids = available_ships[request.ship_id]
        print(f"DEBUG: Tank IDs: {tank_ids}")
        ship_params = get_ship_parameters(request.ship_id)
        print(f"DEBUG: Ship params: {ship_params}")

        # Optional vessel-level temperature/pressure correction tables
        try:
            ship_data_dir = os.path.join(ship_dir, request.ship_id)
            tempcorr_path = os.path.join(ship_data_dir, f"tempcorr_table_{tank_ids[0]}.csv")
            presscorr_path = os.path.join(ship_data_dir, f"presscorr_table_{tank_ids[0]}.csv")

            tempcorr_df = pd.read_csv(tempcorr_path) if os.path.exists(tempcorr_path) else None
            presscorr_df = pd.read_csv(presscorr_path) if os.path.exists(presscorr_path) else None

            def interpolate_value(df: pd.DataFrame, x_col: str, y_col: str, x_value: float) -> float:
                interp_func = RegularGridInterpolator((df[x_col].values,), df[y_col].values)
                return float(interp_func([[x_value]])[0])

            def get_temp_corr(temp_value: float) -> float:
                return 1.0 if tempcorr_df is None else interpolate_value(tempcorr_df, "Temp", "tcorr", temp_value)

            def get_press_corr(press_value: float) -> float:
                return 1.0 if presscorr_df is None else interpolate_value(presscorr_df, "Press", "pcorr", press_value)

        except Exception as e:
            print(f"DEBUG: Failed to initialize correction tables, defaulting to 1.0: {e}")

            def get_temp_corr(temp_value: float) -> float:  # type: ignore
                return 1.0

            def get_press_corr(press_value: float) -> float:  # type: ignore
                return 1.0

        # Parse times
        try:
            opening_datetime = datetime.strptime(request.opening_time, "%m/%d/%Y %H:%M")
            closing_datetime = datetime.strptime(request.closing_time, "%m/%d/%Y %H:%M")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid time format. Use MM/DD/YYYY HH:MM")
        if closing_datetime <= opening_datetime:
            raise HTTPException(status_code=400, detail="Closing time must be after opening time")

        difference_in_hours = (closing_datetime - opening_datetime).total_seconds() / 3600.0
        print(f"DEBUG: Time difference: {difference_in_hours} hours")

        # Opening tank 1
        print("DEBUG: Computing corrected values for opening tank 1 ...")
        cl1, cv1 = compute_corrected_values(
            request.ship_id,
            tank_ids[0],
            request.opening_tank1.level,
            request.opening_list,
            request.opening_trim,
            request.opening_tank1.vapor_temp,
            request.opening_tank1.pressure,
        )
        print(f"DEBUG: Corrected level1: {cl1}, volume1: {cv1}")
        if cl1 is None or cv1 is None:
            raise HTTPException(status_code=500, detail="Failed to compute corrected values for opening tank 1")

        temp_corr1 = get_temp_corr(request.opening_tank1.liquid_temp)
        press_corr1 = get_press_corr(request.opening_tank1.pressure)
        liquid_volume1 = cv1 * temp_corr1 * press_corr1
        vap_corr1 = (273 + 15) / (273 + request.opening_tank1.vapor_temp) * (1.013 + request.opening_tank1.pressure) / 1.013 * 0.6785
        vnet1 = ship_params["LNG_TK1_cap"] - liquid_volume1
        vnet_corr1 = vnet1 * vap_corr1
        vap_volume1 = vnet_corr1 / request.density / 1000.0
        total_volume1 = float(liquid_volume1 + vap_volume1)

        # Opening tank 2 (if exists)
        total_volume2 = 0.0
        if len(tank_ids) > 1 and request.opening_tank2:
            print("DEBUG: Computing corrected values for opening tank 2 ...")
            cl2, cv2 = compute_corrected_values(
                request.ship_id,
                tank_ids[1],
                request.opening_tank2.level,
                request.opening_list,
                request.opening_trim,
                request.opening_tank2.vapor_temp,
                request.opening_tank2.pressure,
            )
            print(f"DEBUG: Corrected level2: {cl2}, volume2: {cv2}")
            if cl2 is not None and cv2 is not None:
                temp_corr2 = get_temp_corr(request.opening_tank2.liquid_temp)
                press_corr2 = get_press_corr(request.opening_tank2.pressure)
                liquid_volume2 = cv2 * temp_corr2 * press_corr2
                vap_corr2 = (273 + 15) / (273 + request.opening_tank2.vapor_temp) * (1.013 + request.opening_tank2.pressure) / 1.013 * 0.6785
                vnet2 = ship_params.get("LNG_TK2_cap", ship_params["LNG_TK1_cap"]) - liquid_volume2
                vnet_corr2 = vnet2 * vap_corr2
                vap_volume2 = vnet_corr2 / request.density / 1000.0
                total_volume2 = float(liquid_volume2 + vap_volume2)

        grand_total_volume_opening = total_volume1 + total_volume2
        print(f"DEBUG: Grand total opening: {grand_total_volume_opening}")

        # Closing tank 1
        print("DEBUG: Computing corrected values for closing tank 1 ...")
        cl3, cv3 = compute_corrected_values(
            request.ship_id,
            tank_ids[0],
            request.closing_tank1.level,
            request.closing_list,
            request.closing_trim,
            request.closing_tank1.vapor_temp,
            request.closing_tank1.pressure,
        )
        print(f"DEBUG: Corrected level3: {cl3}, volume3: {cv3}")
        if cl3 is None or cv3 is None:
            raise HTTPException(status_code=500, detail="Failed to compute corrected values for closing tank 1")

        temp_corr3 = get_temp_corr(request.closing_tank1.liquid_temp)
        press_corr3 = get_press_corr(request.closing_tank1.pressure)
        liquid_volume3 = cv3 * temp_corr3 * press_corr3
        vap_corr3 = (273 + 15) / (273 + request.closing_tank1.vapor_temp) * (1.013 + request.closing_tank1.pressure) / 1.013 * 0.6785
        vnet3 = ship_params["LNG_TK1_cap"] - liquid_volume3
        vnet_corr3 = vnet3 * vap_corr3
        vap_volume3 = vnet_corr3 / request.density / 1000.0
        total_volume3 = float(liquid_volume3 + vap_volume3)

        # Closing tank 2 (if exists)
        total_volume4 = 0.0
        if len(tank_ids) > 1 and request.closing_tank2:
            print("DEBUG: Computing corrected values for closing tank 2 ...")
            cl4, cv4 = compute_corrected_values(
                request.ship_id,
                tank_ids[1],
                request.closing_tank2.level,
                request.closing_list,
                request.closing_trim,
                request.closing_tank2.vapor_temp,
                request.closing_tank2.pressure,
            )
            print(f"DEBUG: Corrected level4: {cl4}, volume4: {cv4}")
            if cl4 is not None and cv4 is not None:
                temp_corr4 = get_temp_corr(request.closing_tank2.liquid_temp)
                press_corr4 = get_press_corr(request.closing_tank2.pressure)
                liquid_volume4 = cv4 * temp_corr4 * press_corr4
                vap_corr4 = (273 + 15) / (273 + request.closing_tank2.vapor_temp) * (1.013 + request.closing_tank2.pressure) / 1.013 * 0.6785
                vnet4 = ship_params.get("LNG_TK2_cap", ship_params["LNG_TK1_cap"]) - liquid_volume4
                vnet_corr4 = vnet4 * vap_corr4
                vap_volume4 = vnet_corr4 / request.density / 1000.0
                total_volume4 = float(liquid_volume4 + vap_volume4)

        grand_total_volume_closing = total_volume3 + total_volume4
        print(f"DEBUG: Grand total closing: {grand_total_volume_closing}")

        # Final results
        vol_diff = grand_total_volume_closing - grand_total_volume_opening
        bog_cons = (request.bog * difference_in_hours / request.density) / 1000.0
        loaded_qty = vol_diff + bog_cons
        total_loaded_qty = loaded_qty + request.unreckoned_qty
        net_qty = request.net_energy / (request.gross_energy / request.bdn_quantity)
        diff = total_loaded_qty - net_qty

        print("=== DEBUG: Calculation completed successfully ===")
        return BunkeringResponse(
            ship_id=request.ship_id,
            tank1_volume_opening=round(total_volume1, 2),
            tank2_volume_opening=round(total_volume2, 2) if len(tank_ids) > 1 else None,
            tank1_volume_closing=round(total_volume3, 2),
            tank2_volume_closing=round(total_volume4, 2) if len(tank_ids) > 1 else None,
            opening_quantity=round(grand_total_volume_opening, 2),
            closing_quantity=round(grand_total_volume_closing, 2),
            volume_difference=round(vol_diff, 2),
            bog_consumption=round(bog_cons, 2),
            loaded_quantity=round(total_loaded_qty, 2),
            net_quantity=round(net_qty, 2),
            difference=round(diff, 2),
            calculation_time=datetime.now().isoformat(),
        )

    except Exception as e:
        print("=== DEBUG: Error occurred ===")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
