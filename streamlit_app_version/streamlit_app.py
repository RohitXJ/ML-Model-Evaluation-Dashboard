import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import prep_data
from pipeline import design_pipeline
from sklearn.linear_model import LogisticRegression

# Store session state
if 'X' not in st.session_state:
    st.session_state.X = None
    st.session_state.y = None
    st.session_state.sorted_cols = None
    st.session_state.sorted_scores = None
    st.session_state.feature_selection_ready = False

MODEL_OPTIONS = {
    1: "Logistic Regression",
    2: "Random Forest",
    3: "SVM",
    4: "Naive Bayes",
    5: "Gradient Boosting"
}

def plot_importance(columns, scores, top_n=10):
    # Convert to array (if not already)
    columns = np.array(columns)
    scores = np.array(scores)

    # Sort in descending order of importance
    sorted_indices = np.argsort(scores)[::-1]
    sorted_columns = columns[sorted_indices]
    sorted_scores = scores[sorted_indices]

    # Only take top_n
    top_columns = sorted_columns[:top_n]
    top_scores = sorted_scores[:top_n]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_columns[::-1], top_scores[::-1], color="skyblue")
    ax.set_xlabel("Importance Score")
    ax.set_title("Top Feature Importances")
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="ML Model Evaluation", layout="wide")
    st.title("🧠 ML Model Evaluation Dashboard")

    uploaded_file = st.file_uploader("📁 Upload your CSV dataset", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("✅ Dataset uploaded successfully!")

            # --- Select target column and drop columns ---
            target_column = st.selectbox("🎯 Select Target Column", data.columns)
            all_columns = [col for col in data.columns if col != target_column]
            columns_to_drop = st.multiselect("🛠️ Drop Columns (Optional if unknown)", all_columns)

            selected_model_names = st.multiselect(
                "📊 Pick one or more models to evaluate",
                options=list(MODEL_OPTIONS.keys()),
                format_func=lambda x: MODEL_OPTIONS[x]
            )

            if st.button("🚀 Run Evaluation"):
                if not selected_model_names:
                    st.warning("⚠️ Please select at least one model to evaluate.")
                    return

                try:
                    X, y = prep_data(data, target_column, columns_to_drop)
                    st.session_state.X = X
                    st.session_state.y = y

                    model = LogisticRegression(max_iter=1000, solver="liblinear")
                    model.fit(X, y)

                    importances = np.abs(model.coef_[0])
                    sorted_idx = np.argsort(importances)[::-1]
                    st.session_state.sorted_cols = X.columns[sorted_idx]
                    st.session_state.sorted_scores = importances[sorted_idx]

                    st.success("✅ Data Preprocessed & Importance Calculated")
                    st.session_state.feature_selection_ready = True

                except Exception as e:
                    st.error("❌ Error during preprocessing or feature importance calculation.")
                    st.exception(e)

        except Exception as e:
            st.error("❌ Could not read uploaded CSV.")
            st.exception(e)

    # --- Feature Selection Section ---
    if st.session_state.feature_selection_ready:
        st.subheader("📈 Feature Importance Plot")

        # Always show the plot again if already computed
        plot_importance(
            st.session_state.sorted_cols,
            st.session_state.sorted_scores,
            top_n=15
        )

        st.markdown("---")
        st.subheader("🧮 Select Feature Selection Method")

        method = st.radio("Choose one:", ["Auto (elbow method)", "Manual (top-N)"])
        selected_top_n = None

        if method == "Manual (top-N)":
            max_n = len(st.session_state.sorted_cols)
            selected_top_n = st.selectbox(
                "🔢 Select number of top features",
                list(range(1, max_n + 1)),
                index=min(9, max_n - 1)
            )

        if st.button("✅ Proceed with Feature Selection and Model Evaluation"):
            try:
                X, y = st.session_state.X, st.session_state.y
                sorted_cols = st.session_state.sorted_cols
                sorted_scores = st.session_state.sorted_scores

                if method == "Manual (top-N)":
                    X = X[sorted_cols[:selected_top_n]]
                    st.success(f"✅ Using top {selected_top_n} features.")
                elif method == "Auto (elbow method)":
                    diffs = np.diff(sorted_scores)
                    elbow = np.argmax(diffs < 0.01) + 1
                    elbow = max(elbow, 1)
                    X = X[sorted_cols[:elbow]]
                    st.success(f"✅ Auto-selected top {elbow} features using elbow method.")

                # Model Evaluation
                st.info("🔍 Evaluating models...")
                results = design_pipeline(X, y, selected_model_names)

                st.subheader("📊 Results")
                for result in results:
                    with st.expander(f"📌 {result['name']}", expanded=True):
                        st.markdown(result["evaluation"])
                        st.pyplot(result["fig"])

                st.balloons()

            except Exception as e:
                st.error("❌ Error during model evaluation.")
                st.exception(e)

if __name__ == "__main__":
    main()
