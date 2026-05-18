import io
import re
import difflib
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# =========================
# Streamlit 基础设置
# =========================
st.set_page_config(
    page_title="NailVesta 截图识别英文名对照表",
    page_icon="💅",
    layout="wide",
)

st.title("💅 NailVesta 截图 → SKU / 英文名 / 中文名对照表")
st.caption("上传“名称”截图 + 产品图册 CSV，程序会 OCR 识别截图里的中文名，并去图册里匹配 SKU、英文名、中文名。")


# =========================
# 工具函数
# =========================
def read_csv_from_bytes(file_bytes: bytes) -> Tuple[pd.DataFrame, str]:
    """尽量兼容 UTF-8 / GBK 等 CSV 编码。"""
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "big5", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            return df, enc
        except Exception as e:
            last_error = e

    raise RuntimeError(f"CSV 读取失败，请检查文件格式。最后错误：{last_error}")


def normalize_text(value) -> str:
    """用于匹配的清洗：去空格、标点、表头干扰，但保留中文/英文/数字。"""
    if pd.isna(value):
        return ""

    s = str(value).strip()
    s = s.replace("\n", "").replace("\r", "").replace("\t", "")
    s = re.sub(r"\s+", "", s)

    # 去掉 OCR 可能识别出来的表头/无关词
    s = s.replace("名称", "")
    s = s.replace("中文名称", "")
    s = s.replace("款式名称", "")
    s = s.replace("Name", "")
    s = s.replace("name", "")

    # 保留中英文、数字、常见符号
    s = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\-_./（）()&+#]", "", s)
    return s.strip()


def has_chinese(value) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(value)))


def infer_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """根据关键词自动猜列名。"""
    for key in candidates:
        for col in columns:
            if key.lower() in str(col).lower():
                return col
    return None


def parse_lines(text: str) -> List[str]:
    """把 OCR / 手动输入的内容拆成中文名列表。"""
    lines = []
    for raw in text.splitlines():
        cleaned = normalize_text(raw)
        if not cleaned:
            continue
        if cleaned in {"名称", "中文名称", "款式名称", "Name"}:
            continue
        # 截图里一般都是中文款式名；太短的噪音跳过
        if len(cleaned) < 2:
            continue
        lines.append(cleaned)

    # 保留顺序去重
    deduped = []
    seen = set()
    for item in lines:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


@st.cache_resource(show_spinner=False)
def load_easyocr_reader():
    """加载 EasyOCR。第一次会比较慢，之后会缓存。"""
    import easyocr
    return easyocr.Reader(["ch_sim", "en"], gpu=False)


def run_ocr(image_bytes: bytes) -> Tuple[List[str], List[dict]]:
    """OCR 识别截图，按从上到下排序。"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 截图通常很窄，放大后中文识别更稳
    w, h = img.size
    scale = 3 if w < 800 else 2
    img = img.resize((w * scale, h * scale))

    arr = np.array(img)
    reader = load_easyocr_reader()

    results = reader.readtext(
        arr,
        detail=1,
        paragraph=False,
        contrast_ths=0.1,
        adjust_contrast=0.7,
        width_ths=0.7,
    )

    # EasyOCR 返回：[bbox, text, confidence]
    rows = []
    for item in results:
        bbox, text, conf = item
        y_top = min([p[1] for p in bbox])
        rows.append({"y": y_top, "text": text, "confidence": float(conf)})

    rows = sorted(rows, key=lambda x: x["y"])
    raw_lines = [r["text"] for r in rows]
    cleaned_lines = parse_lines("\n".join(raw_lines))
    return cleaned_lines, rows


def fuzzy_match(query: str, choices: List[str]) -> Tuple[Optional[str], float]:
    """优先用 rapidfuzz；没有安装时用 difflib 兜底。"""
    try:
        from rapidfuzz import process, fuzz

        match = process.extractOne(query, choices, scorer=fuzz.WRatio)
        if match is None:
            return None, 0
        best_choice, score, _ = match
        return best_choice, float(score)
    except Exception:
        best = difflib.get_close_matches(query, choices, n=1, cutoff=0)
        if not best:
            return None, 0
        score = difflib.SequenceMatcher(None, query, best[0]).ratio() * 100
        return best[0], float(score)


def build_mapping_table(
    names: List[str],
    catalog_df: pd.DataFrame,
    cn_col: str,
    en_col: str,
    sku_col: Optional[str],
    shape_col: Optional[str],
    location_col: Optional[str],
    threshold: int,
) -> pd.DataFrame:
    """把截图中文名匹配到图册，生成英文名中文名对照表。"""
    df = catalog_df.copy()

    df["_cn_norm"] = df[cn_col].apply(normalize_text)
    df = df[df["_cn_norm"] != ""].copy()

    # 如果中文名重复，保留第一条，但后续结果备注会提示可能重复
    duplicate_norms = set(df[df["_cn_norm"].duplicated(keep=False)]["_cn_norm"].tolist())

    exact_map = {}
    for _, row in df.iterrows():
        norm = row["_cn_norm"]
        if norm not in exact_map:
            exact_map[norm] = row

    choices = df["_cn_norm"].dropna().astype(str).unique().tolist()

    output_rows = []

    for original_name in names:
        query_norm = normalize_text(original_name)

        matched_row = None
        matched_norm = None
        match_type = ""
        score = 0.0

        if query_norm in exact_map:
            matched_row = exact_map[query_norm]
            matched_norm = query_norm
            match_type = "精确匹配"
            score = 100.0
        else:
            best_norm, best_score = fuzzy_match(query_norm, choices)
            if best_norm and best_score >= threshold:
                matched_row = exact_map.get(best_norm)
                matched_norm = best_norm
                match_type = "模糊匹配"
                score = best_score

        if matched_row is None:
            output_rows.append({
                "English Name": "",
                "中文名称": "",
                "SKU": "",
                "甲型": "",
                "库位": "",
                "截图识别文本": original_name,
                "图册匹配中文名": "",
                "匹配方式": "未匹配",
                "匹配分数": round(score, 1),
                "备注": "未在产品图册中找到，请检查 OCR 是否识别错字，或图册是否缺少该款式。",
            })
            continue

        cn_value = str(matched_row.get(cn_col, "")).strip()
        en_value_raw = matched_row.get(en_col, "")
        en_value = "" if pd.isna(en_value_raw) else str(en_value_raw).strip()

        remarks = []
        if matched_norm in duplicate_norms:
            remarks.append("图册里该中文名存在重复，请人工确认 SKU。")
        if not en_value:
            remarks.append("图册英文名为空。")
        elif has_chinese(en_value) and normalize_text(en_value) == normalize_text(cn_value):
            remarks.append("图册英文名栏疑似仍是中文，建议补英文名。")

        output_rows.append({
            "English Name": en_value,
            "中文名称": cn_value,
            "SKU": "" if not sku_col or pd.isna(matched_row.get(sku_col, "")) else str(matched_row.get(sku_col, "")).strip(),
            "甲型": "" if not shape_col or pd.isna(matched_row.get(shape_col, "")) else str(matched_row.get(shape_col, "")).strip(),
            "库位": "" if not location_col or pd.isna(matched_row.get(location_col, "")) else str(matched_row.get(location_col, "")).strip(),
            "截图识别文本": original_name,
            "图册匹配中文名": cn_value,
            "匹配方式": match_type,
            "匹配分数": round(score, 1),
            "备注": "；".join(remarks),
        })

    return pd.DataFrame(output_rows)



# =========================
# 上传区
# =========================
left, right = st.columns([1, 1])

with left:
    catalog_file = st.file_uploader(
        "1️⃣ 上传产品图册 CSV",
        type=["csv"],
        help="需要包含中文名称列和款式英文名称列。",
    )

with right:
    image_files = st.file_uploader(
        "2️⃣ 上传名称截图（可多张）",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="上传类似只有“名称”这一列的截图。",
    )

manual_text_default = ""
catalog_df = None

if catalog_file is not None:
    try:
        catalog_df, detected_encoding = read_csv_from_bytes(catalog_file.getvalue())
        st.success(f"产品图册读取成功：{len(catalog_df)} 行，编码：{detected_encoding}")

        all_cols = list(catalog_df.columns)

        default_cn = infer_column(all_cols, ["中文名称", "中文", "名称"])
        default_en = infer_column(all_cols, ["款式英文名称", "英文名称", "英文", "English"])
        default_sku = infer_column(all_cols, ["SKU"])
        default_shape = infer_column(all_cols, ["甲型", "shape"])
        default_location = infer_column(all_cols, ["库位", "location"])

        with st.expander("字段确认 / 修改", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                cn_col = st.selectbox(
                    "图册里的中文名称列",
                    all_cols,
                    index=all_cols.index(default_cn) if default_cn in all_cols else 0,
                )
                en_col = st.selectbox(
                    "图册里的英文名称列",
                    all_cols,
                    index=all_cols.index(default_en) if default_en in all_cols else 0,
                )

            with col2:
                sku_options = ["不输出"] + all_cols
                sku_col_pick = st.selectbox(
                    "SKU 列",
                    sku_options,
                    index=sku_options.index(default_sku) if default_sku in sku_options else 0,
                )
                shape_col_pick = st.selectbox(
                    "甲型列",
                    sku_options,
                    index=sku_options.index(default_shape) if default_shape in sku_options else 0,
                )

            with col3:
                location_col_pick = st.selectbox(
                    "库位列",
                    sku_options,
                    index=sku_options.index(default_location) if default_location in sku_options else 0,
                )
                threshold = st.slider("模糊匹配最低分", min_value=60, max_value=100, value=82, step=1)

        sku_col = None if sku_col_pick == "不输出" else sku_col_pick
        shape_col = None if shape_col_pick == "不输出" else shape_col_pick
        location_col = None if location_col_pick == "不输出" else location_col_pick

    except Exception as e:
        st.error(f"产品图册读取失败：{e}")


# =========================
# OCR 识别区
# =========================
st.divider()
st.subheader("3️⃣ OCR 识别截图 / 手动校正")

if "ocr_text" not in st.session_state:
    st.session_state["ocr_text"] = ""

if image_files:
    with st.expander("截图预览", expanded=False):
        for f in image_files:
            st.image(f, caption=f.name, use_container_width=False)

    if st.button("开始 OCR 识别截图", type="primary"):
        all_names = []
        debug_rows = []

        try:
            with st.spinner("正在识别截图中文名，第一次运行可能需要 1–3 分钟……"):
                for f in image_files:
                    names, rows = run_ocr(f.getvalue())
                    all_names.extend(names)
                    for r in rows:
                        r["file"] = f.name
                    debug_rows.extend(rows)

            # 保留顺序去重
            deduped = []
            seen = set()
            for n in all_names:
                if n not in seen:
                    deduped.append(n)
                    seen.add(n)

            st.session_state["ocr_text"] = "\n".join(deduped)
            st.session_state["ocr_debug"] = pd.DataFrame(debug_rows)

            st.success(f"OCR 完成：识别到 {len(deduped)} 个名称。请在下方检查，错字可以直接改。")
        except ModuleNotFoundError:
            st.error(
                "当前环境没有安装 easyocr。请在 requirements.txt 加上 easyocr、opencv-python-headless、numpy、pillow，"
                "或者先把截图里的中文名手动粘贴到下方文本框。"
            )
        except Exception as e:
            st.error(f"OCR 识别失败：{e}")
            st.info("你也可以先把截图里的中文名复制/手动输入到下方文本框，一行一个。")

manual_text = st.text_area(
    "OCR 识别结果 / 手动输入区（一行一个中文名，可直接修改错字）",
    value=st.session_state.get("ocr_text", ""),
    height=180,
    placeholder="例如：\n白法中方\n玫花长方\n橘贝壳长方",
)

with st.expander("查看 OCR 原始识别明细", expanded=False):
    debug_df = st.session_state.get("ocr_debug")
    if isinstance(debug_df, pd.DataFrame) and not debug_df.empty:
        st.dataframe(debug_df, use_container_width=True)
    else:
        st.caption("还没有 OCR 明细。")


# =========================
# 匹配输出区
# =========================
st.divider()
st.subheader("4️⃣ 生成 SKU / 英文名 / 中文名对照表")

if st.button("生成对照表"):
    if catalog_df is None:
        st.error("请先上传产品图册 CSV。")
        st.stop()

    names = parse_lines(manual_text)
    if not names:
        st.error("没有可匹配的中文名。请先 OCR 截图，或在文本框手动输入中文名。")
        st.stop()

    result_df = build_mapping_table(
        names=names,
        catalog_df=catalog_df,
        cn_col=cn_col,
        en_col=en_col,
        sku_col=sku_col,
        shape_col=shape_col,
        location_col=location_col,
        threshold=threshold,
    )

    matched_df = result_df[result_df["匹配方式"] != "未匹配"].copy()
    unmatched_df = result_df[result_df["匹配方式"] == "未匹配"].copy()

    simple_df = matched_df[["SKU", "English Name", "中文名称"]].copy()

    st.success(f"完成：共 {len(result_df)} 个名称，成功匹配 {len(matched_df)} 个，未匹配 {len(unmatched_df)} 个。")

    st.markdown("#### 简版对照表：SKU / 英文名 / 中文名")
    st.dataframe(simple_df, use_container_width=True)

    st.markdown("#### 完整追溯表")
    st.dataframe(result_df, use_container_width=True)

    if not unmatched_df.empty:
        st.warning("以下名称未匹配，请检查截图识别错字或产品图册是否缺少这些款式：")
        st.dataframe(unmatched_df[["截图识别文本", "备注"]], use_container_width=True)

    needs_en_fix = result_df[result_df["备注"].astype(str).str.contains("英文名", na=False)].copy()
    if not needs_en_fix.empty:
        st.info("以下款式的英文名栏可能需要在图册里补充/修正：")
        st.dataframe(needs_en_fix[["中文名称", "English Name", "SKU", "备注"]], use_container_width=True)

    simple_csv_bytes = simple_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下载简版 CSV：SKU / 英文名 / 中文名",
        data=simple_csv_bytes,
        file_name="sku_english_chinese_mapping.csv",
        mime="text/csv",
    )

    full_csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下载完整追溯 CSV",
        data=full_csv_bytes,
        file_name="sku_english_chinese_mapping_full_trace.csv",
        mime="text/csv",
    )
