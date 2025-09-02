# Imports and Configuration
from enum import Enum
import glob
import json
import os
import re
import warnings
import locale
import numpy as np
import pandas as pd
import numexpr as ne

from typing import Annotated, List, Dict, Union, List, Literal
from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request, Response, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from datetime import datetime, time, timedelta
from routers.session_manager import sessions, connections
# Custom local imports
from common import initlog
from Visualization import visualization_page
from mysql_insertion import mysql_insertion

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


class Table(str, Enum):
    label = "label"
    grade = "grading"

TABLE_COLS = {
    Table.label: label_numeric_cols,
    Table.grade: grading_numeric_cols,
}


def get_file(
    table : str
):
    if table.value == 'label':
        file = '已登錄產品之細項與核准項目 (溫熱型開飲機_ALL)-110_標註顯示欄位.xlsx'
    else:
        file = '溫熱型開飲機節能標章產品規格(含108-110年產銷量)_標註必要欄位.xlsx'

    df = pd.read_excel(f'D:/markding_git/itri_etl_analysis/api_code/uploaded_files/{file}')

    logger.info(f'df : {df.head()}')

    return df


@router.get("/sales_diff", response_class=HTMLResponse)
def get_file_types(
    table :str = Table,
    percent: Annotated[float, Query(gt=0, lt=100)] = 10,
    year: Annotated[int, Query(gt=2000, lt=2100)] = datetime.now().year,
):

    df = get_file(table)
    pass


@router.get("/options/{table}")
def get_options(table: Table):
    """Frontend calls this to populate the dropdowns."""
    return {"numeric_cols": TABLE_COLS[table]}


@router.get("/get_bubble_chart/")
def get_chart_data(
    table: Table = Query(..., description="label or grading"),
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


class PreviewReq(BaseModel):
    table: Table
    sales_col: str = Field(..., description="銷售量欄位")
    benchmark_col: str = "column_name"
    benchmark_col2: str = "column_name2"
    formula: str = Field(..., description="以 ${欄位名} 為佔位的算式")
    params: Dict[str, str] = Field(default_factory=dict, description="語意名->實際欄位")
    output_col: str = "節電結果"
    sample: int = 10

    @field_validator("sales_col")
    @classmethod
    def sales_in_allowed(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("sales_col 無效")
        return v

SAFE_FUNCS = {"min": min, "max": max, "abs": abs}

# # ---- helpers ----
# def placeholder_to_var(name: str) -> str:
#     return "col_" + re.sub(r"[^0-9a-zA-Z_]", "_", name)

def compile_formula(formula: str):
    used = []
    col = formula.group(1)
    used.append(col)
    return  used


def guard_expression(expr: str):
    banned = ["__","import","eval","exec","os.","sys.","open("]
    if any(x in expr for x in banned):
        raise HTTPException(400, "公式包含不允許內容")


@router.post("/energy/preview")
def energy_preview(req: PreviewReq):
    """
    Example:
    {
    "table":"label",
    "sales_col":"109銷售量",
    "benchmark_col":{"type":"value","value":1000},
    "formula":"(${sales} - ${benchmark_col}) * ${保溫功率(W)} / 1000",
    "params":{"功率W":"保溫功率(W)"},
    "output_col":"節電結果_kWh",
    "sample":5
    }   
    """

    try:
        print(req.benchmark_col)
        df = get_file(req.table)
        
        need = {req.sales_col, req.benchmark_col, *req.params.values()}
        # print('need========', need)
        # print('df.columns========', df.columns)

        missing = [c for c in need if c not in df.columns]
        if missing:
            raise HTTPException(400, f"缺少欄位: {missing}")

        # used = compile_formula(req.formula)
        
        print('compiled',req.formula)
        guard_expression(req.formula)

        # 建立變數環境
        env = {}
        # 標準變數：sales / benchmark
        env["sales_col"] = df[req.sales_col].fillna(0.0).to_list()
        env[("benchmark_col")] = df[req.benchmark_col].fillna(0.0).to_list()
        env[("benchmark_col2")] = df[req.benchmark_col2].fillna(0.0).to_list()
        
        # for col in set([req.sales_col, req.benchmark_col, *req.params.values(), *used]):
        #     if col in df.columns:
        #         env[placeholder_to_var(col)] = df[col]
        # 額外別名映射（如 ${保溫功率(W)}）
        # for alias, real_col in req.params.items():
        #     env[placeholder_to_var(alias)] = df[real_col]

        # # NA policy
        # if req.na_policy == "fill0":
        #     for k in list(env.keys()):
        #         s = env[k]
        #         if isinstance(s, pd.Series):
        #             env[k] = s.fillna(0)

        print('env==========',env)
        print('req.formula==========',req.formula)
        # 計算
        try:
            result = ne.evaluate(req.formula, local_dict=env)
        except Exception as e:
            raise HTTPException(400, f"公式解析/計算失敗: {e}")

        df['power_saving_result'] = result

        row_count = mysql_insertion(df, f'power_saving_result_{req.table.value}')

        if row_count:
            return JSONResponse(content= {'result': f"data saved row_count : [{row_count}]"})
        # 彙總與預覽

    except Exception as e:
        logger.error(f'Something wrong {e}', exc_info=True)


{
  "table":"label",
  "sales_col":"109銷售量",
  "benchmark_col":{"type":"value","name":"溫熱型開飲機節能標章能源耗用基準(E)(kWh/24h)"},
  "formula":"(${sales} - ${benchmark_col}) * ${保溫功率(W)} / 1000",
  "params":{"功率W":"保溫功率(W)"},
  "output_col":"節電結果_kWh",
  "sample":5
}
