# Same as app.py; provided as a separate filename to avoid browser/download cache confusion.
import io
import re
import difflib
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

APP_VERSION = "v2026-05-18-no-screenshot"

st.set_page_config(
    page_title="NailVesta 中文名匹配英文名",
    page_icon="💅",
    layout="wide",
)

st.title("💅 NailVesta 中文名 → SKU / 英文名 / 中文名对照表")
st.caption(f"{APP_VERSION}｜这个版本不需要上传截图，只需要上传产品图册 CSV，并粘贴中文名。")


def read_csv_safely(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """读取 CSV，自动尝试常见编码。"""
    file_bytes = uploaded_file.getvalue()
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "big5", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            df.columns = [str(c).strip() for c in df.columns]
            return df, enc
        except Exception as e:
            last_error = e

    raise RuntimeError(f"CSV 读取失败，请确认文件是 CSV 格式。最后错误：{last_error}")


def normalize_text(value) -> str:
    """匹配用清洗：去空格、换行、隐藏字符、常见标点。"""
    if pd.isna(value):
        return ""
    s = str(value)
    s = s.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", "")
    s = s.strip().replace("\n", "").replace("\r", "").replace("\t", "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\-_./（）()&+#]", "", s)
    return s.strip()


def has_chinese(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(value)))


def infer_column(columns: List[str], keywords: List[str]) -> Optional[str]:
    for key in keywords:
        for col in columns:
            if key.lower() in str(col).lower():
                return col
    return None


def parse_names(text: str) -> List[str]:
    """支持一行一个，也支持逗号/顿号/分号分隔。"""
    if not text:
        return []

    text = text.replace("，", "\n").replace(",", "\n")
    text = text.replace("、", "\n").replace("；", "\n").replace(";", "\n")

    names = []
    for line in text.splitlines():
        item = normalize_text(line)
        if item and item not in ["名称", "中文名", "中文名称", "款式名称"]:
            names.append(item)

    # 保留顺序去重
    result = []
    seen = set()
    for name in names:
        if name not in seen:
            result.append(name)
            seen.add(name)
    return result


def fuzzy_match(query: str, choices: List[str]) -> Tuple[Optional[str], float]:
    """优先 rapidfuzz；没有安装时用 difflib。"""
    try:
        from rapidfuzz import process, fuzz
        match = process.extractOne(query, choices, scorer=fuzz.WRatio)
        if not match:
            return None, 0.0
        return str(match[0]), float(match[1])
    except Exception:
        matches = difflib.get_close_matches(query, choices, n=1, cutoff=0)
        if not matches:
            return None, 0.0
        score = difflib.SequenceMatcher(None, query, matches[0]).ratio() * 100
        return matches[0], score


def build_result(
    catalog_df: pd.DataFrame,
    pasted_names: List[str],
    sku_col: str,
    cn_col: str,
    en_col: str,
    threshold: int,
) -> pd.DataFrame:
    df = catalog_df.copy()
    df["_cn_norm"] = df[cn_col].apply(normalize_text)
    df = df[df["_cn_norm"] != ""].copy()

    exact_map = {}
    for _, row in df.iterrows():
        # 如果图册同一个中文名重复，先保留第一条，并在备注提示
        exact_map.setdefault(row["_cn_norm"], row)

    duplicate_names = set(df[df["_cn_norm"].duplicated(keep=False)]["_cn_norm"].tolist())
    choices = list(exact_map.keys())

    rows = []
    for i, raw_name in enumerate(pasted_names, start=1):
        query = normalize_text(raw_name)
        matched_row = None
        match_type = "未匹配"
        score = 0.0
        matched_norm = ""

        if query in exact_map:
            matched_row = exact_map[query]
            matched_norm = query
            match_type = "精确匹配"
            score = 100.0
        else:
            best, best_score = fuzzy_match(query, choices)
            if best and best_score >= threshold:
                matched_row = exact_map[best]
                matched_norm = best
                match_type = "模糊匹配"
                score = best_score

        if matched_row is None:
            rows.append({
                "序号": i,
                "SKU": "",
                "英文名": "",
                "中文名": raw_name,
                "输入中文名": raw_name,
                "图册匹配中文名": "",
                "匹配方式": "未匹配",
                "匹配分数": round(score, 1),
                "备注": "未在产品图册找到，请检查错字或确认图册是否包含该款。",
            })
            continue

        sku = "" if pd.isna(matched_row.get(sku_col, "")) else str(matched_row.get(sku_col, "")).strip()
        cn = "" if pd.isna(matched_row.get(cn_col, "")) else str(matched_row.get(cn_col, "")).strip()
        en = "" if pd.isna(matched_row.get(en_col, "")) else str(matched_row.get(en_col, "")).strip()

        notes = []
        if matched_norm in duplicate_names:
            notes.append("图册里该中文名重复，请人工确认 SKU")
        if not en:
            notes.append("图册英文名为空")
        elif has_chinese(en) and normalize_text(en) == normalize_text(cn):
            notes.append("图册英文名栏疑似仍是中文，建议补英文名")

        rows.append({
            "序号": i,
            "SKU": sku,
            "英文名": en,
            "中文名": cn,
            "输入中文名": raw_name,
            "图册匹配中文名": cn,
            "匹配方式": match_type,
            "匹配分数": round(score, 1),
            "备注": "；".join(notes),
        })

    return pd.DataFrame(rows)


# 1. 上传产品图册
st.subheader("1️⃣ 上传产品图册 CSV")
catalog_file = st.file_uploader(
    "产品图册 CSV",
    type=["csv"],
    help="需要包含 SKU、中文名称、款式英文名称等字段。",
)

catalog_df = None
sku_col = cn_col = en_col = None
threshold = 82

if catalog_file is not None:
    try:
        catalog_df, encoding = read_csv_safely(catalog_file)
        st.success(f"产品图册读取成功：{len(catalog_df)} 行｜编码：{encoding}")

        columns = list(catalog_df.columns)
        default_sku = infer_column(columns, ["SKU", "sku", "货号", "编码"])
        default_cn = infer_column(columns, ["中文名称", "中文名", "中文", "名称"])
        default_en = infer_column(columns, ["款式英文名称", "英文名称", "英文名", "英文", "English", "name"])

        with st.expander("字段确认", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                sku_col = st.selectbox(
                    "SKU 列",
                    columns,
                    index=columns.index(default_sku) if default_sku in columns else 0,
                )
            with c2:
                cn_col = st.selectbox(
                    "中文名列",
                    columns,
                    index=columns.index(default_cn) if default_cn in columns else 0,
                )
            with c3:
                en_col = st.selectbox(
                    "英文名列",
                    columns,
                    index=columns.index(default_en) if default_en in columns else 0,
                )
            with c4:
                threshold = st.slider("模糊匹配最低分", 60, 100, 82, 1)

        preview_cols = [sku_col, en_col, cn_col]
        preview_cols = list(dict.fromkeys([c for c in preview_cols if c]))
        st.markdown("**产品图册预览**")
        st.dataframe(catalog_df[preview_cols].head(20), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"产品图册读取失败：{e}")


# 2. 粘贴中文名
st.divider()
st.subheader("2️⃣ 粘贴中文名")
pasted_text = st.text_area(
    "把中文名粘贴在这里，一行一个",
    height=180,
    placeholder="白法中方\n玫花长方\n橘贝壳长方",
)

pasted_names = parse_names(pasted_text)
if pasted_text:
    st.caption(f"已识别到 {len(pasted_names)} 个中文名。")


# 3. 生成结果
st.divider()
st.subheader("3️⃣ 生成对照表")

if st.button("生成 SKU / 英文名 / 中文名对照表", type="primary"):
    if catalog_df is None:
        st.error("请先上传产品图册 CSV。")
        st.stop()

    if not pasted_names:
        st.error("请先粘贴中文名。")
        st.stop()

    result_df = build_result(
        catalog_df=catalog_df,
        pasted_names=pasted_names,
        sku_col=sku_col,
        cn_col=cn_col,
        en_col=en_col,
        threshold=threshold,
    )

    simple_df = result_df[["SKU", "英文名", "中文名"]].copy()
    matched_count = int((result_df["匹配方式"] != "未匹配").sum())
    unmatched_count = int((result_df["匹配方式"] == "未匹配").sum())

    st.success(f"完成：共 {len(result_df)} 个，成功匹配 {matched_count} 个，未匹配 {unmatched_count} 个。")

    st.markdown("#### 简版对照表")
    st.dataframe(simple_df, use_container_width=True, hide_index=True)

    st.download_button(
        "下载简版 CSV",
        data=simple_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="sku_english_chinese_mapping.csv",
        mime="text/csv",
    )

    st.markdown("#### 完整追溯表")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    st.download_button(
        "下载完整追溯 CSV",
        data=result_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="sku_english_chinese_mapping_full.csv",
        mime="text/csv",
    )

    unmatched_df = result_df[result_df["匹配方式"] == "未匹配"]
    if not unmatched_df.empty:
        st.warning("以下中文名未匹配：")
        st.dataframe(unmatched_df[["中文名", "备注"]], use_container_width=True, hide_index=True)
