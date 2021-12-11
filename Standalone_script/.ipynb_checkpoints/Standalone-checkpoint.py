import json
import logging
import os
import traceback
import webbrowser

import folium
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from branca.element import Template, MacroElement
from win32api import MessageBox


def prepare_test_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    A function to calculate the new columns from the provided data in the form of pandas dataframe.
    Columns like WindThreshold, GustTheshold, etc will be added based on the calculations and dataframe will be
    prepared for the model. Thresholds column will be removed from the dataframe.

    :param data: Dataframe to used to add and remove columns.
    :type data: dataframe
    :return: Updated dataframe ready to be used for model
    :rtype: dataframe
    """
    data[["WindThreshold", "GustTheshold"]] = data.Thresholds.str.split('/', expand=True)
    data["WindThreshold"] = data["WindThreshold"].astype(float)
    data["GustTheshold"] = data["GustTheshold"].astype(float)
    data = data[data["WindThreshold"] > 0][data["GustTheshold"] > 0]
    data["closeToWindThreshold"] = round((data["WindSustained"] * 100) / data["WindThreshold"].astype(float), 2)
    data["closeToGustThreshold"] = round((data["GustSustained"] * 100) / data["GustTheshold"].astype(float), 2)

    data["eFPI"] = data["FPI"].apply(lambda x: round(x / fpi_thres, 2))
    data["eThres"] = round(data["WindSustained"] / data["WindThreshold"], 2)
    data["eThres"] = np.where(round(data["GustSustained"] / data["GustTheshold"], 2) > data["eThres"],
                              round(data["GustSustained"] / data["GustTheshold"], 2), data["eThres"])

    data.drop(["Thresholds"], axis=1, inplace=True)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)
    return data


def fnc_category(x: int) -> str:
    """
    A function to return the category of the provided threshold value.
    Currently divide into 3 categories.

    :param x: Value of the threshold to be divide into categorical value
    :type x: int
    :return: Returned value of category of the provided threshold
    :rtype: str
    """
    if x > pred_thrs_high:
        return "C"
    elif pred_thrs_high >= x > pred_thrs_low:
        return "B"
    else:
        return "A"


def highlight_cell_excel(s: pd.core.series.Series) -> list:
    """
    A function to add the cell background color based on the predicted column value

    :param s: Series to be used for the background color
    :type s: int
    :return: Returned value of the color coding of the row in the form of list
    :rtype: list
    """

    is_max = pd.Series(data=False, index=s.index)
    is_max[y_colmn_predctd] = s.loc[y_colmn_predctd] > pred_thrs_high
    if is_max.any():
        return ['background-color: #FF0000' for v in is_max]
    is_max[y_colmn_predctd] = s.loc[y_colmn_predctd] > pred_thrs_low
    if is_max.any():
        return ['background-color: #FFFF00' for v in is_max]
    else:
        return ['background-color: #00FF00' for v in is_max]


def predict_values():
    """
    A function to predict the values of the provided csv by the user. It will save the csv in the folder mentioned
    in the config file.
    """
    main_df = pd.read_csv(input_csv_pth)
    if main_df.empty:
        print('Dataset is empty!')
        raise ("Empty input dataset")
    save_excel_path = f"{save_path[:-3]}xlsx"
    logger.info("Preparing data for the prediction")
    test_df_raw = prepare_test_data(main_df)
    test_df = test_df_raw[x_columns]
    logger.info("Prediction dataset prepared")

    logger.info(f"Loading model from the mentioned pkl file path: {model_path}")
    modll = joblib.load(model_path)
    logger.info("Predicting values for the input dateset")
    y_predctd = modll.predict(test_df)
    logger.info("Prediction completed")
    y_predctd = pd.DataFrame(y_predctd, columns=[y_colmn_predctd])
    ds = pd.concat([test_df_raw, y_predctd], axis=1)
    ds[y_colmn_predctd] = round(ds[y_colmn_predctd] * 100, 2)

    ds.to_csv(save_path, index=False)
    print(f"CSV Report generated at: {save_path}")
    # ds = ds.sort_values(by=[y_colmn_predctd], ascending=False)
    print("Generating Excel report...")
    ds.style.apply(highlight_cell_excel, axis=1).to_excel(save_excel_path, engine='xlsxwriter', index=False)
    logger.info(f"File saved to mentioned path:\nCSV path: {save_path}\nExcel path: {save_excel_path}")
    print(f"Excel Report generated at: {save_excel_path}")


def show_categorical_data():
    """
    A function to display the number of High, Medium, Low risk points over the graph.
    """
    logger.info("Reading CSV")
    fire_grp = pd.read_csv(save_path)

    msg = f"{y_colmn_predctd} Range:\nHigh (RED): {y_colmn_predctd} > {pred_thrs_high}" \
          f"\nMedium (YELLOW): {pred_thrs_high} >= {y_colmn_predctd} > {pred_thrs_low}\n" \
          f"Low (GREEN): {y_colmn_predctd}<={pred_thrs_low}"

    logger.info(msg)
    fire_grp["pspsGRP"] = fire_grp[y_colmn_predctd].apply(lambda x: fnc_category(x))
    month = fire_grp.groupby(['pspsGRP']).size().reset_index(name='count')[::-1].sort_values('pspsGRP')

    values = month['count'].tolist()
    if len(values) < 3:
        values += [0] * (3 - len(values))

    plt.figure(figsize=(8, 6), dpi=100)
    plt.ylabel('Predicted Incidents', size=15)
    plt.xlabel('Severity', size=15)
    plt.bar(labels, values, color=['green', 'yellow', 'red'])
    xlocs = [0, 1, 2]
    for i, v in enumerate(month['count']): plt.text(xlocs[i] - 0.05, v + 5, str(v))

    colors = {'High (Probability>85)': 'red', 'Medium (85>=Probability>60)': 'yellow', 'Low (Probability<=60)': 'green'}
    labelsss = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[labelllll]) for labelllll in labelsss]
    plt.legend(handles, labelsss)
    print(msg)

    plt.show()


def generate_map():
    """
    A fucntion to generate map from the predicted report
    """
    map_dta = pd.read_csv(save_path)
    map_dta['Circuit'] = map_dta['Circuit'].str.upper().str.strip()
    geom_dct = json.load(open("json\circuit_geom.json", 'r'))
    idx = map_dta.groupby(['Circuit'])[y_colmn_predctd].transform(max) == map_dta[y_colmn_predctd]
    map_dta = map_dta[idx].reset_index(drop=True)

    def categorize_dta(flnm, dtaaaaa):
        circuit_dct = {}
        lst = []
        for index, row in dtaaaaa.iterrows(): circuit_dct[row['Circuit']] = row.to_dict()
        for i in geom_dct['features']:
            tmp = circuit_dct.get(i["properties"]["CIRCUIT_NAME"].upper(), -99)
            if (tmp != -99):
                i["properties"]["FPI"] = tmp["FPI"]
                i["properties"]["WindSustained"] = tmp["WindSustained"]
                i["properties"]["GustSustained"] = tmp["GustSustained"]
                i["properties"]["WindThreshold"] = tmp["WindThreshold"]
                i["properties"]["GustTheshold"] = tmp["GustTheshold"]
                i["properties"]["PSPS_Activation_probability%"] = tmp["PSPS_Activation_probability%"]
                lst += [i]
        dct = {"type": "FeatureCollection", "features": lst}
        json.dump(dct, open(flnm, 'w'))

    green_dta = map_dta[60 >= map_dta['PSPS_Activation_probability%']]
    orange_dta = map_dta[
        (85 >= map_dta['PSPS_Activation_probability%']) & (map_dta['PSPS_Activation_probability%'] > 60)]
    red_dta = map_dta[map_dta['PSPS_Activation_probability%'] > 85]
    categorize_dta(json_green, green_dta)
    categorize_dta(json_yellow, orange_dta)
    categorize_dta(json_red, red_dta)
    m = folium.Map(
        location=[34.39, -118.55],
        tiles="cartodbpositron",
        zoom_start=10,
    )

    for tile in tiles: folium.TileLayer(tile).add_to(m)

    folium.GeoJson(json_green, name="Low (Probability<=60)",
                   style_function=lambda x: {'color': 'green', 'highlight': 'True',
                                             'line_weight': 2},
                   popup=folium.GeoJsonPopup(fields=map_popup_fields, aliases=map_popup_fields_alias)).add_to(m)

    folium.GeoJson(json_yellow, name="Medium (85>=Probability>60)", style_function=lambda x: {'color': 'yellow',
                                                                                              'highlight': 'True',
                                                                                              'line_weight': 2},
                   popup=folium.GeoJsonPopup(fields=map_popup_fields, aliases=map_popup_fields_alias)).add_to(m)

    folium.GeoJson(json_red, name="High (Probability>85)", style_function=lambda x: {'color': 'red',
                                                                                     'highlight': 'True',
                                                                                     'line_weight': 2},
                   popup=folium.GeoJsonPopup(fields=map_popup_fields, aliases=map_popup_fields_alias)).add_to(m)

    folium.LayerControl().add_to(m)

    macro = MacroElement()
    macro._template = Template(open(legend_template_path).read())
    m.get_root().add_child(macro)
    m.save(map_save_path)


try:
    # Initializing the parameters from the Standalone_config.json (config) file
    input_csv_pth = input("Path of input csv: ")
    cnfg = json.load(open("Standalone_config.json"))
    x_columns = cnfg["x_columns"]
    model_path = cnfg["model_path"]
    save_path = cnfg["save_path"]
    y_colmn_predctd = cnfg["y_colmn_predctd"]
    labels = cnfg["labels"]
    err_log_pth = cnfg["err_log_pth"]
    err_log_format = cnfg["err_log_format"]
    fpi_thres = cnfg["fpi_thres"]
    pred_thrs_low, pred_thrs_high = cnfg["pred_thrs_range"]
    json_green = cnfg['json_green']
    json_yellow = cnfg['json_yellow']
    json_red = cnfg['json_red']
    tiles = cnfg['tiles']
    map_popup_fields = cnfg['map_popup_fields']
    map_popup_fields_alias = cnfg['map_popup_fields_alias']
    map_save_path = cnfg['map_save_path']
    legend_template_path = cnfg['legend_template_path']

    logging.basicConfig(filename=err_log_pth, filemode='a+', format=err_log_format)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("Starting application")
    print("Analysing input csv, please wait...")
    predict_values()
    print("Model successfully executed")
    logger.info("Displaying Categorical graph")
    show_categorical_data()

    if MessageBox(None, "Do you want to generate Map from the predicted result?", "Confirmation", 1) == 1:
        print("Generating Map...")
        generate_map()
        webbrowser.open('file://' + os.path.realpath(map_save_path))

except:
    logger.error(traceback.format_exc())
    print(f"Some error occurred, please check mentioned error log file: {err_log_pth}")
finally:
    input("\nPress Enter to exit")
