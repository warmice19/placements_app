import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt


st.title("PG Placements ")

df = pd.read_csv("./../data/placements_data.csv")

# -----------------------------------------------------
# Load Model and Utilities
# -----------------------------------------------------

model = pickle.load(open("./../model/model_logistic_regression.pkl", "rb"))
ss_input_vars = pickle.load(open("./../model/ss_input_vars.pkl", "rb"))
ohe_input_vars = pickle.load(open("./../model/ohe_input_vars.pkl", "rb"))
le_target_vars = pickle.load(open("./../model/le_target_vars.pkl", "rb"))


df = df.rename(
    columns={
        "ssc_p": "10th_percentage",
        "ssc_b": "10th_board",
        "hsc_p": "12th_percentage",
        "hsc_b": "12th_board",
        "hsc_s": "stream",
        "degree_p": "undergrad_percentage",
        "degree_t": "undergrad_stream",
        "mba_p": "mba_percentage",
    }
)

df.drop("sl_no", axis=1, inplace=True)

nav = st.sidebar.selectbox("Navigate", ["Plots", "Check your Chances"])

# ------------------------------
# PLACEMENT PROBABILITY
# ------------------------------
if nav == "Check your Chances":
    # -----------------------------------------------------
    # Get User Input
    # -----------------------------------------------------
    st.sidebar.header("Enter your details:")

    def user_input():
        tenth_p = st.sidebar.slider("10th Percentage: ", 0, 100, 50)
        twelfth_p = st.sidebar.slider("12th Percentage: ", 0, 100, 50)
        ug_p = st.sidebar.slider("UG Percentage: ", 0, 100, 50)
        mba_p = st.sidebar.slider("MBA Percentage: ", 0, 100, 50)
        etest_p = st.sidebar.slider("Placement Test Percentage: ", 0, 100, 50)
        gender_ip = st.sidebar.radio("Gender", ("M", "F"))
        tenth_board_ip = st.sidebar.selectbox("10th Board", ("Central", "Others"))
        twelfth_board_ip = st.sidebar.selectbox("12th Board", ("Central", "Others"))
        twelfth_stream = st.sidebar.selectbox(
            "12th Stream", ("Science", "Commerce", "Arts")
        )
        UG_stream = st.sidebar.selectbox(
            "UG Stream", ("Sci&Tech", "Comm&Mgmt", "Others")
        )
        workex_ip = st.sidebar.radio("Work Experience", ("Yes", "No"))
        MBA_stream_ip = st.sidebar.selectbox(
            "MBA Specialization", ("Mkt&HR", "Mkt&Fin")
        )

        # etest_p = st.sidebar.slider("Placement Test Percentage: ", 0, 100, 50)

        return {
            "10th_percentage": tenth_p,
            "12th_percentage": twelfth_p,
            "undergrad_percentage": ug_p,
            "mba_percentage": mba_p,
            "etest_p": etest_p,
            "gender": gender_ip,
            "10th_board": tenth_board_ip,
            "12th_board": twelfth_board_ip,
            "stream": twelfth_stream,
            "undergrad_stream": UG_stream,
            "workex": workex_ip,
            "specialisation": MBA_stream_ip,
        }

    user_input_dict = user_input()

    # Convertig user input dictionary to lists
    user_input_numerical = [v for i, v in enumerate(user_input_dict.values()) if i <= 4]
    user_input_categorical = [
        v for i, v in enumerate(user_input_dict.values()) if i > 4
    ]

    # Converting user input list to numpy array and reshape. Reshape is required because
    # number of rows should be equal to one as we are running predcitions on a single
    # input row
    arr_user_input_numerical = np.array(user_input_numerical).reshape(1, -1)
    arr_user_input_categorical = np.array(user_input_categorical).reshape(1, -1)

    # One Hot Encoding categorical features array.
    # Normalising numerical features list using standard scaler.
    # Both one hot encoder and the standard scaler were created in the training set
    arr_user_input_numerical = ss_input_vars.transform(arr_user_input_numerical)
    arr_user_input_categorical = ohe_input_vars.transform(arr_user_input_categorical)

    # Joining the two lists
    arr_user_input = np.hstack((arr_user_input_numerical, arr_user_input_categorical.A))

    # -----------------------------------------------------
    # Generate Predictions
    # -----------------------------------------------------
    pred = model.predict_proba(arr_user_input)

    # st.write(le_target_vars.classes_)
    pred_np = pred[0, 0] * 100
    pred_p = pred[0, 1] * 100
    coeff_model = model.coef_[0]
    coeff_bar = px.bar([x for x in range(len(coeff_model))], coeff_model)
    if pred_p > pred_np:
        st.markdown(
            """
        ## **Congratulations!!**
        """
        )
        st.write("You have a " + "{:.2f}".format(pred_p) + " chance of getting placed")
        st.markdown(
            """
            _To check the impact of various features in you result, click on the button below_
        """
        )
        if st.button("Check"):
            st.write(coeff_bar)

    if pred_p < pred_np:
        st.markdown(
            """
        ## Not there yet
        ### Keep Trying
        """
        )
        st.write(
            "You have only " + "{:.2f}".format(pred_p) + " chance of  getting placed"
        )
        st.write(
            "To check the relative effect of various factors, click on the button below"
        )

        if st.button("Check"):
            st.write(coeff_bar)


# ------------------------------
# PLOTS
# ------------------------------
if nav == "Plots":
    st.markdown(
        """
    ## Dataframe.Describe()
    _Details of the dataset._
    """
    )
    st.write(df)
    st.markdown(
        """
    ## Dataframe.Describe()
    _Basic statistical details of the dataset._
    """
    )
    st.write(df.describe())

    # --------
    # HEATMAP
    # --------
    st.markdown(
        """
    ## heatmap().
    _Correlation between features of the dataset._
    """
    )
    plt.style.use("dark_background")
    hmap = plt.figure()
    sns.heatmap(df.corr(), annot=True)
    st.pyplot(hmap)

    # -----------
    # COUNTPLOTS
    # -----------

    st.markdown(
        """ ## Countplot()
    _Placements counts for different categorical data._"""
    )
    (
        gender_beta,
        tenth_beta,
        twelfth_beta,
        twelfth_stream_beta,
        ugStream_beta,
        workex,
        mba_spec_beta,
    ) = st.beta_columns(7)

    if gender_beta.button("Gender"):
        plotly_counplt = px.histogram(df, x="status", color="gender")
        st.write(plotly_counplt)

    if tenth_beta.button("10th Board"):
        plotly_counplt = px.histogram(df, x="status", color="10th_board")
        st.write(plotly_counplt)

    if twelfth_beta.button("12th Board"):
        plotly_counplt = px.histogram(df, x="status", color="12th_board")
        st.write(plotly_counplt)

    if twelfth_stream_beta.button("12th Stream"):
        plotly_counplt = px.histogram(df, x="status", color="stream")
        st.write(plotly_counplt)

    if ugStream_beta.button("UG Stream"):
        plotly_counplt = px.histogram(df, x="status", color="undergrad_stream")
        st.write(plotly_counplt)

    if workex.button("Work Experience"):
        plotly_counplt = px.histogram(df, x="status", color="workex")
        st.write(plotly_counplt)

    if mba_spec_beta.button("MBA Specialisation"):
        plotly_counplt = px.histogram(df, x="status", color="specialisation")
        st.write(plotly_counplt)

    # -------------------
    # PLOTLY HISTOGRAMS
    # --------------------
    st.markdown(
        """
    ## Histogram()
    _Distribution of numeric data in the form of histogram._"""
    )

    tenth_p, twelfth_p, ug_p, mba_p, test_p = st.beta_columns([1, 1, 1, 1, 1])

    if tenth_p.button("10th Percentage"):
        plotly_hist = px.histogram(df, x="10th_percentage", nbins=50)
        st.write(plotly_hist)

    if twelfth_p.button("12th Percentage"):
        plotly_hist = px.histogram(df, x="12th_percentage", nbins=50)
        st.write(plotly_hist)

    if ug_p.button("UG Percentage"):
        plotly_hist = px.histogram(df, x="undergrad_percentage", nbins=50)
        st.write(plotly_hist)

    if mba_p.button("MBA Percentage"):
        plotly_hist = px.histogram(df, x="mba_percentage", nbins=50)
        st.write(plotly_hist)

    if test_p.button("Placement Test Percentage"):
        plotly_hist = px.histogram(df, x="etest_p", nbins=50)
        st.write(plotly_hist)

    # ----------------------------------------------------
    # COUNTPLOTS: Salary breakup for the complete season
    # ----------------------------------------------------
    st.markdown(
        """ ## Salary breakup for the placement duration.
    _Placement offers for the particular college campus._
    """
    )
    Salcountplt = plt.figure()
    ax = sns.countplot(data=df, x=df["salary"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=80, Fontsize=8)
    plt.tight_layout()
    st.pyplot(Salcountplt)

    # -----------------------------------
    # BOXPLOTS: Gender wise performance
    # -----------------------------------
    st.markdown(
        """ 
    ## Boxplot().
    _Gender wise difference in performance over the years._"""
    )
    tenth_box, twelfth_box, ug_box, etest_box, mba_box = st.beta_columns(
        [1, 1, 1, 1, 1]
    )

    if tenth_box.button("10th Performace"):
        plotly_box = px.box(df, x="10th_percentage", y="gender", orientation="h")
        st.write(plotly_box)
    if twelfth_box.button("12th Performace"):
        plotly_box = px.box(df, x="12th_percentage", y="gender", orientation="h")
        st.write(plotly_box)

    if ug_box.button("UG Performace"):
        plotly_box = px.box(df, x="undergrad_percentage", y="gender", orientation="h")
        st.write(plotly_box)

    if etest_box.button("Placement Test Performace"):
        plotly_box = px.box(df, x="etest_p", y="gender", orientation="h")
        st.write(plotly_box)

    if mba_box.button("MBA Performace"):
        plotly_box = px.box(df, x="mba_percentage", y="gender", orientation="h")
        st.write(plotly_box)
