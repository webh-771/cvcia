import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Minimal Data Dashboard", layout="wide")
st.title("Minimal Data Dashboard")

if "df" not in st.session_state:
    st.session_state.df = None

@st.cache_data
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

@st.cache_data
def to_xml_text(df: pd.DataFrame) -> str:
    return df.to_xml(index=False)

def to_pickle_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_pickle(buf)
    return buf.getvalue()

st.sidebar.title("File Operations")
file_type = st.sidebar.selectbox("Format", ["CSV", "Pickle", "XML"], key="fmt")
mode = st.sidebar.radio("Action", ["Read", "Write", "Append"], key="mode")

def read_any(uploaded, fmt):
    if fmt == "CSV":
        return pd.read_csv(uploaded)
    elif fmt == "Pickle":
        return pd.read_pickle(uploaded)
    else:
        return pd.read_xml(uploaded)

if mode == "Read":
    uploader = st.sidebar.file_uploader(
        "Select file to READ",
        type={"CSV": ["csv"], "Pickle": ["pkl"], "XML": ["xml"]}[file_type],
        key="read_uploader"
    )
    if st.sidebar.button("Load file", key="btn_read") and uploader:
        try:
            st.session_state.df = read_any(uploader, file_type)
            st.sidebar.success("File loaded.")
        except Exception as e:
            st.sidebar.error(f"Read failed: {e}")

elif mode == "Append":
    uploader = st.sidebar.file_uploader(
        "Select file to APPEND",
        type={"CSV": ["csv"], "Pickle": ["pkl"], "XML": ["xml"]}[file_type],
        key="append_uploader"
    )
    if st.sidebar.button("Append to data", key="btn_append"):
        if st.session_state.df is None:
            st.sidebar.error("No base data loaded; read a file first.")
        elif not uploader:
            st.sidebar.error("Select a file to append.")
        else:
            try:
                extra = read_any(uploader, file_type)
                st.session_state.df = pd.concat([st.session_state.df, extra], ignore_index=True)
                st.sidebar.success("Appended file data.")
            except Exception as e:
                st.sidebar.error(f"Append failed: {e}")

else:  
    if st.session_state.df is None:
        st.sidebar.info("Load data to enable downloads.")
    else:
        df = st.session_state.df
        st.sidebar.download_button(
            "Download CSV",
            data=to_csv_bytes(df),
            file_name="data.csv",
            mime="text/csv",
            key="dl_csv",
            use_container_width=True
        )
        st.sidebar.download_button(
            "Download Pickle",
            data=to_pickle_bytes(df),
            file_name="data.pkl",
            mime="application/octet-stream",
            key="dl_pkl",
            use_container_width=True
        )
        st.sidebar.download_button(
            "Download XML",
            data=to_xml_text(df),
            file_name="data.xml",
            mime="application/xml",
            key="dl_xml",
            use_container_width=True
        )

if st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.write("Summary:")
    try:
        st.dataframe(df.describe(include="all"), use_container_width=True)
    except Exception:
        st.info("Summary not available for this data.")

    cols = st.multiselect("Convert columns", df.columns, key="convert_cols")
    type_opt = st.selectbox("To type", ["int", "float", "str", "datetime"], key="convert_type")
    if st.button("Convert", key="btn_convert"):
        changed = False
        for c in cols:
            try:
                if type_opt == "datetime":
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                else:
                    df[c] = df[c].astype(type_opt)
                changed = True
            except Exception:
                pass
        if changed:
            st.session_state.df = df
            st.success("Conversion done.")
        else:
            st.warning("No changes applied.")

    st.write("---")
    st.write("Basic Data Operations")
    op = st.selectbox("Operation", ["Sort", "Group", "Slice", "Filter"], key="op_select")

    if op == "Sort":
        sc = st.selectbox("Column", df.columns, key="sort_col")
        if st.button("Sort", key="btn_sort"):
            try:
                st.dataframe(df.sort_values(by=sc), use_container_width=True)
            except Exception as e:
                st.error(f"Sort failed: {e}")

    elif op == "Group":
        gc = st.multiselect("Group by", df.columns, key="group_cols")
        agg = st.selectbox("Aggregation", ["mean", "count", "sum", "median"], key="group_agg")
        if st.button("Group", key="btn_group"):
            if not gc:
                st.warning("Select at least one group-by column.")
            else:
                try:
                    st.dataframe(df.groupby(gc).agg(agg), use_container_width=True)
                except Exception as e:
                    st.error(f"Group failed: {e}")

    elif op == "Slice":
        nrows = len(df)
        start = st.number_input("Start", min_value=0, max_value=max(0, nrows - 1), value=0, key="slice_start")
        end_default = min(5, nrows) if nrows else 0
        end = st.number_input("End", min_value=0, max_value=nrows, value=end_default, key="slice_end")
        if st.button("Slice", key="btn_slice"):
            try:
                st.dataframe(df.iloc[int(start):int(end)], use_container_width=True)
            except Exception as e:
                st.error(f"Slice failed: {e}")

    else: 
        fc = st.selectbox("Filter column", df.columns, key="filter_col")
        val = st.text_input("Equals value", key="filter_val")
        if st.button("Filter", key="btn_filter"):
            try:
                st.dataframe(df[df[fc].astype(str) == val], use_container_width=True)
            except Exception as e:
                st.error(f"Filter failed: {e}")

    st.write("---")
    st.subheader("Charts & Numpy")
    nc = df.select_dtypes(include=[np.number]).columns.tolist()
    plot = st.selectbox("Plot Type", ["Bar", "Line", "Histogram"], key="plot_type")

    if plot in ["Bar", "Line"] and len(nc):
        x = st.selectbox("X", df.columns, key="sel_x")
        y = st.selectbox("Y (numeric)", nc, key="sel_y")
        if x and y:
            data = df[[x, y]].dropna()
            if plot == "Bar":
                try:
                    st.bar_chart(data, x=x, y=y, use_container_width=True)
                except TypeError:
                    st.bar_chart(data.set_index(x)[y], use_container_width=True)
            else:
                try:
                    st.line_chart(data, x=x, y=y, use_container_width=True)
                except TypeError:
                    st.line_chart(data.set_index(x)[y], use_container_width=True)

    elif plot == "Histogram" and len(nc):
        n = st.selectbox("Numeric column", nc, key="sel_hist")
        if n:
            fig, ax = plt.subplots()
            ax.hist(df[n].dropna(), bins=30)
            ax.set_xlabel(n)
            ax.set_ylabel("Count")
            st.pyplot(fig)

    if len(nc):
        st.write("Mean:", float(df[nc].mean().mean()))
        st.write("Median:", float(df[nc].median().median()))
        st.write("Std:", float(df[nc].std().mean()))
else:
    st.info("Load a file to begin.")
