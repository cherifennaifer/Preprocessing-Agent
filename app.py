import streamlit as st
import pandas as pd
from utils.preprocessing import load_data
from crew_pipeline import build_preprocessing_plan, run_pipeline
from tasks.analyze_data import analyze_dataset

st.title("AI Preprocessing Agent")

# --- Step Tracker ---
def show_steps(current_step):
    steps = [
        "Step 1: Upload dataset",
        "Step 2: Preview data",
        "Step 3: Summary",
        "Step 4: Handle missing values",
        "Step 5: Run pipeline",
        "Step 6: Cleaning report & download"
    ]
    progress_value = current_step / len(steps)
    st.progress(progress_value)
    for i, step in enumerate(steps, start=1):
        if i == current_step:
            st.markdown(f"➡️ **{step}**")
        elif i < current_step:
            st.markdown(f"✅ {step}")
        else:
            st.markdown(f"⚪ {step}")


def invalidate_confirmation():
    st.session_state.confirmed_changes = False


def reset_workflow():
    st.session_state.current_step = 1
    for key in [
        "df", "summary", "missing_info", "df_clean", "report",
        "pipeline_summary", "pipeline_advice", "advice_cache",
        "advice_model_index", "confirmed_changes", "remove_duplicates",
        "normalize_data", "advisor_chat", "summary_signature",
    ]:
        st.session_state.pop(key, None)


@st.dialog("Reset workflow")
def confirm_reset_dialog():
    st.write("This will clear the uploaded dataset, preprocessing plan, and results.")
    confirm_col, cancel_col = st.columns(2)
    if confirm_col.button("Confirm reset", type="primary", use_container_width=True):
        reset_workflow()
        st.rerun()
    if cancel_col.button("Cancel", use_container_width=True):
        st.rerun()

# --- Initialize step state ---
if "current_step" not in st.session_state:
    st.session_state.current_step = 1
if "missing_info" not in st.session_state:
    st.session_state.missing_info = []
if "confirmed_changes" not in st.session_state:
    st.session_state.confirmed_changes = False
if "remove_duplicates" not in st.session_state:
    st.session_state.remove_duplicates = True
if "normalize_data" not in st.session_state:
    st.session_state.normalize_data = True
if "advisor_chat" not in st.session_state:
    st.session_state.advisor_chat = []

show_steps(st.session_state.current_step)

# --- Reset Workflow ---
if st.sidebar.button("🔄 Reset Workflow"):
    confirm_reset_dialog()


# --- Step 1: Upload ---
if st.session_state.current_step == 1:
    uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv","xlsx","json"])
    if uploaded_file:
        st.session_state.current_step = 2

# --- Step 2: Preview ---
if st.session_state.current_step >= 2 and "df" not in st.session_state:
    if uploaded_file.name.endswith(".csv"):
        st.session_state.df = load_data(uploaded_file, "csv")
    elif uploaded_file.name.endswith(".xlsx"):
        st.session_state.df = load_data(uploaded_file, "xlsx")
    elif uploaded_file.name.endswith(".json"):
        st.session_state.df = load_data(uploaded_file, "json")

if st.session_state.current_step == 2:
    st.write("### 📄 Preview of Data")
    st.dataframe(st.session_state.df.head())
    st.session_state.current_step = 3

# --- Step 3: Summary ---
# Keep the summary visible during Step 4 so toggle interactions do not hide it.
if st.session_state.current_step in (3, 4):
    summary = analyze_dataset(st.session_state.df)
    summary_signature = (
        summary["rows"],
        tuple(summary["column_names"]),
        tuple(summary["dtypes"].items()),
        tuple(summary["missing_values"].items()),
    )
    if st.session_state.get("summary_signature") != summary_signature:
        for key in list(st.session_state):
            if key.startswith("action_") or key.startswith("approve_"):
                del st.session_state[key]
        st.session_state.confirmed_changes = False
        st.session_state.missing_info = []
        st.session_state.summary_signature = summary_signature
    st.session_state.summary = summary

    # Global infos
    st.markdown("### 📊 Dataset Summary")
    st.markdown(f"- **Rows:** `{summary['rows']}`")
    st.markdown(f"- **Columns:** `{summary['columns']}`")
    st.markdown(f"- **Duplicate Rows:** `{summary['duplicate_count']}`")

    # Missing values global count
    total_missing = sum(summary["missing_values"].values())
    st.markdown(f"- **Total Missing Values:** `{total_missing}`")

    # Toggle for advanced details
    show_details = st.toggle("Show advanced details (table + chart)")
    if show_details:
        # Table
        table_data = []
        for col in summary["dtypes"].keys():
            table_data.append([
                col,
                summary["dtypes"][col],
                summary["missing_values"].get(col, 0)
            ])
        st.write("### 🧾 Column Overview")
        st.dataframe(pd.DataFrame(table_data, columns=["Column", "Type", "Missing Values"]))

        # Bar chart of missing values
        st.write("### 📉 Missing Values per Column")
        missing_df = pd.DataFrame.from_dict(summary["missing_values"], orient="index", columns=["Missing Values"])
        st.bar_chart(missing_df)

    st.markdown("### 🤖 Preprocessing Guide")
    st.info(
        f"""
        🗑️ **Duplicates**  
        - {summary['duplicate_count']} duplicate rows  
        - {"✅ No action required" if summary['duplicate_count'] == 0 else "⚠️ Consider removing duplicates"}

        🧩 **Missing Values**  
        - Total missing entries: {sum(summary['missing_values'].values())}  
        - 👉 Small gaps → impute with median/mode  
        - ⚠️ High-missing columns (>70%) → suggest dropping after review  

        📏 **Normalization**  
        - Numeric columns → scale (StandardScaler / MinMaxScaler)  
        - Categorical columns → encode (one-hot or target encoding)
        """
    )

    # Quick Checklist
    st.markdown("### ✅ Quick Action Checklist")
    st.success(
        """
        - 📅 Parse `date_added` → datetime (set NaT for invalids)  
        - 🧩 Fill missing text values with `Unknown` without adding columns  
        - 🗑️ Drop columns with >70% missing unless essential (e.g., imdb_id, tmdb_id, imdb_votes, tmdb_popularity)  
        - 🏷️ Convert categorical columns to category dtype (rating, primary_genre, primary_country)  
        - 📏 Scale numeric features if using distance‑based or linear models  
        """
    )
    st.session_state.current_step = 4



# --- Step 4: Missing Values ---
if st.session_state.current_step == 4:
    if "summary" not in st.session_state:
        st.error("⚠️ No dataset summary found. Please go back to Step 3 first.")
    else:
        summary = st.session_state.summary
        missing_info = build_preprocessing_plan(summary)
        action_options = [
            "preserve", "drop_row", "median", "mean", "mode", "unknown",
            "datetime", "drop", "skip",
        ]

        st.markdown("### ⚙️ Review preprocessing plan")
        st.caption("Rules are suggestions. Edit any action to match your domain knowledge.")
        for item in missing_info:
            column = item["column"]
            action_key = f"action_{column}"
            approval_key = f"approve_{column}"
            if action_key not in st.session_state:
                st.session_state[action_key] = item["method"]
            elif st.session_state[action_key] == "unknown_with_flag":
                st.session_state[action_key] = "unknown"
            if approval_key not in st.session_state:
                st.session_state[approval_key] = item["approved"]

            st.selectbox(
                f"Action for {column} ({item['missing']} missing, {item['dtype']})",
                action_options,
                key=action_key,
                on_change=invalidate_confirmation,
            )
            st.checkbox(
                f"Approve change for {column}",
                key=approval_key,
                on_change=invalidate_confirmation,
            )
            st.caption(item["explanation"])
            item["method"] = st.session_state[action_key]
            item["approved"] = st.session_state[approval_key]

        st.session_state.missing_info = missing_info
        if "remove_duplicates" not in st.session_state:
            st.session_state.remove_duplicates = summary["duplicate_count"] > 0
        st.checkbox(
            "Remove duplicate rows",
            key="remove_duplicates",
            on_change=invalidate_confirmation,
        )
        st.checkbox(
            "Normalize numeric feature columns",
            key="normalize_data",
            on_change=invalidate_confirmation,
        )

        if missing_info:
            st.dataframe(pd.DataFrame(
                [[item["column"], item["dtype"], item["missing"], item["method"], item["approved"]] for item in missing_info],
                columns=["Column", "Type", "Missing", "Action", "Approved"],
            ), hide_index=True)
        else:
            st.success("No missing values were found. Review duplicate removal and normalization below.")
        st.info(
            "Rules protect ID columns, convert date columns, use median for numeric measurements, "
            "and add missing flags for text/categorical columns. Nothing is applied until confirmation."
        )

        if st.button("✅ Confirm final plan"):
            st.session_state.confirmed_changes = True
            st.session_state.current_step = 5
            st.rerun()

# --- Step 5: Run pipeline ---
if st.session_state.current_step == 5:
    if "df" not in st.session_state or "summary" not in st.session_state:
        st.error("Please upload a dataset and complete the Summary step first.")
    elif not st.session_state.confirmed_changes:
        st.warning("Review and confirm the final preprocessing plan in Step 4 first.")
    else:
        df_clean, pipeline_summary, advice, report = run_pipeline(
            st.session_state.df,
            st.session_state.missing_info,
            normalize_data=st.session_state.normalize_data,
            remove_dupes=st.session_state.remove_duplicates,
        )
        # ✅ Store results in session_state
        st.session_state.df_clean = df_clean
        st.session_state.pipeline_summary = pipeline_summary
        st.session_state.pipeline_advice = advice
        st.session_state.report = report
        st.session_state.current_step = 6
        st.rerun()


# --- Step 6: Report & Download ---
if st.session_state.current_step == 6:
    report = st.session_state.report
    st.markdown("### 📑 Cleaning Report")

    filled_count = sum(1 for action in report.values() if "Filled" in action)
    dropped_count = sum(1 for action in report.values() if "Dropped" in action)
    normalized = any("Normalization" in action for action in report.values())

    summary_line = f"🧾 Summary: 🧩 {filled_count} filled | 🗑️ {dropped_count} dropped"
    if normalized:
        summary_line += " | 📏 normalization applied"
    st.markdown(f"**{summary_line}**")

    for col, action in report.items():
        if "Dropped" in action:
            icon, color = "🗑️", "red"
        elif "Filled" in action:
            icon, color = "🧩", "green"
        elif "Normalization" in col or "Normalization" in action:
            icon, color = "📏", "blue"
        else:
            icon, color = "⚪", "gray"
        st.markdown(f"{icon} **{col}**: <span style='color:{color}'>{action}</span>", unsafe_allow_html=True)

    st.write("### ✅ Cleaned Data")
    st.dataframe(st.session_state.df_clean.head())

    # ✅ Toggle to compare summaries
    show_pipeline_summary = st.toggle("View Pipeline Summary (compare before vs after)")
    if show_pipeline_summary:
        original = st.session_state.summary
        cleaned = st.session_state.pipeline_summary

        st.write("#### 📊 Before vs After")
        comparison = pd.DataFrame([
            {
                "Metric": "Rows",
                "Before": original["rows"],
                "After": cleaned["rows"],
                "Change": cleaned["rows"] - original["rows"],
            },
            {
                "Metric": "Columns",
                "Before": original["columns"],
                "After": cleaned["columns"],
                "Change": cleaned["columns"] - original["columns"],
            },
            {
                "Metric": "Missing values",
                "Before": sum(original["missing_values"].values()),
                "After": sum(cleaned["missing_values"].values()),
                "Change": sum(cleaned["missing_values"].values()) - sum(original["missing_values"].values()),
            },
            {
                "Metric": "Missing values resolved",
                "Before": "-",
                "After": max(
                    0,
                    sum(original["missing_values"].values())
                    - sum(cleaned["missing_values"].values()),
                ),
                "Change": "-",
            },
            {
                "Metric": "Duplicate rows",
                "Before": original["duplicate_count"],
                "After": cleaned["duplicate_count"],
                "Change": cleaned["duplicate_count"] - original["duplicate_count"],
            },
        ])
        st.dataframe(comparison, hide_index=True)

        st.write("#### Missing Values by Column")
        missing_comparison = pd.DataFrame({
            "Before": pd.Series(original["missing_values"], dtype="int64"),
            "After": pd.Series(cleaned["missing_values"], dtype="int64"),
        }).fillna(0).astype(int)
        st.bar_chart(missing_comparison)

        original_columns = set(original["column_names"])
        cleaned_columns = set(cleaned["column_names"])
        added_columns = sorted(cleaned_columns - original_columns)
        removed_columns = sorted(original_columns - cleaned_columns)
        st.write("#### Column Changes")
        st.dataframe(pd.DataFrame([
            {"Change": "Added", "Columns": ", ".join(added_columns) or "None"},
            {"Change": "Removed", "Columns": ", ".join(removed_columns) or "None"},
        ]), hide_index=True)

    st.sidebar.download_button(
        "Download Cleaned CSV",
        st.session_state.df_clean.to_csv(index=False),
        "cleaned.csv"
    )
