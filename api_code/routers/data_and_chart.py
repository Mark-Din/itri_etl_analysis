# Imports and Configuration
from enum import Enum
import glob
import json
import os
from typing import Annotated, List
from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import locale
from datetime import datetime, time, timedelta
from fastapi import Body, FastAPI
import pandas as pd
import secrets 
from routers.session_manager import sessions, connections
from fastapi import FastAPI, Query, HTTPException
# Custom local imports
from common import initlog
import warnings
from Visualization import visualization_page
from typing import Dict


warnings.filterwarnings("ignore", category=UserWarning)

logger = initlog('fastAPI_data_and_chart')

# Configuration for handling currency formatting in Chinese
locale.setlocale(locale.LC_ALL, 'zh_TW.UTF-8')
router = APIRouter(tags=["data_and_chart"])



grading_numeric_cols = [
    "2020產量", "2020銷售量",
    "額定最大消耗功率(W)", "額定熱水系統消耗功率(W)", "額定保溫加熱器消耗功率(W)",
    "熱水系統貯水桶容量標示值(L)", "熱水系統貯水桶容量實測值(L)",
    "溫水貯水桶容量標示值(L)", "溫水貯水桶容量實測值(L)",
    "熱水系統24小時平均水溫(℃)", "周圍溫度(℃)",
    "每24小時備用損失E24(kWh/24小時)標示值", "每24小時備用損失E24(kWh/24小時)實測值",
    "每24小時標準化備用損失Est,24 (kWh/24小時)標示值",
    "每24小時標準化備用損失Est,24 (kWh/24小時)實測值",
    "每年保溫耗電量"
]
label_numeric_cols = ['108產量',
 '108銷售量',
 '109產量',
 '109銷售量',
 '110產量',
 '110銷售量',
 '寬(mm)',
 '高(mm)',
 '深(mm)',
 '熱水總功率(W)',
 '保溫功率(W)',
 '生水容量(L)',
 '實測熱水貯水桶容量(L)',
 '標示熱水貯水桶容量(L)',
 '溫水箱容量(L)',
 '溫水溫度(℃)',
 '熱水溫度(℃)',
 '熱水系統24小時平均水溫(°C)',
 '周圍溫度(°C)',
 '能源因素值EFtest (L/(kWh/day))',
 '能源因素值EFbase(L/(kWh/day))',
 '每24小時備用損失實測值(E24)( kWh)',
 '等效容積換算係數(K)',
 '每24小時標準化備用損失實測值( Est,24) ( kWh)',
 '每24小時標準化備用損失驗算值( Est,24) ( kWh)',
 '每24小時標準化備用損失標示值( Est,24) ( kWh)',
 '溫熱型開飲機節能標章能源耗用基準(E)(kWh/24h)']



def get_file(
    uplolad_dir: str = './uploaded_files'
):
    file_folders_path = glob.glob(f"{uplolad_dir}/*")
    latest_folder = max(file_folders_path, key=os.path.getmtime)
    files_list = glob.glob(f"{latest_folder}/*")
    logger.info(f'files_list{files_list}')
    if files_list:
        return files_list[0]
    else:
        return None

@router.get("/sales_diff", response_class=HTMLResponse)
def get_file_types(
    uplolad_dir = Depends(get_file),
    percent: Annotated[float, Query(gt=0, lt=100)] = 10,
    year: Annotated[int, Query(gt=2000, lt=2100)] = datetime.now().year,
):

    pass


class Table(str, Enum):
    label = "label"
    grade = "grade"

TABLE_COLS = {
    Table.label: label_numeric_cols,
    Table.grade: grading_numeric_cols,
}

@router.get("/options/{table}")
def get_options(table: Table):
    """Frontend calls this to populate the dropdowns."""
    return {"numeric_cols": TABLE_COLS[table]}


@router.get("/get_bubble_chart/")
def get_chart_data(
    table: Table = Query(..., description="label or grade"),
    x: str = Query(..., description="X-axis column name"),
    y: str = Query(..., description="Y-axis column name"),
):
    allowed = set(TABLE_COLS[table])
    for col in (x, y):
        if col not in allowed:
            raise HTTPException(
                status_code=422,
                detail=f"'{col}' is not valid for table '{table}'. Allowed: {sorted(allowed)}",
            )
    # proceed to build the chart...
    return {"table": table, "x": x, "y": y}

class Symbol(str, Enum):
    addition = '+'
    subtraction = '-'
    multiplication = 'x'
    division = '/'

@router.post('/eff_class_table', response_class=JSONResponse)
def eff_class_table(
    request: Request,
    symbol:Symbol = Query(..., description="label or grade"),
    params: Dict[str, str] = Depends(lambda request: dict(request.query_params))
    ):

    # Extract all query params as a dict
    params = dict(request.query_params)

    # Remove 'symbol' if it’s in params (since already parsed)
    if "symbol" in params:
        params.pop("symbol")

    return {"symbol": symbol, "params": params}