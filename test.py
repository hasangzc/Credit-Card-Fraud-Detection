import xml.etree.ElementTree as ET

import pandas as pd
import requests


def parse_XML(xml_file, df_cols):
    """
    Parse the input XML file and store the result in a pandas
    DataFrame with the given columns.

    The first element of df_cols is supposed to be the identifier
    variable, which is an attribute of each node element in the
    XML data; other features will be parsed from the text content
    of each sub-element.
    """
    # Parse the XML from the string XML
    xroot = ET.fromstring(xml_file)
    rows = []
    # Create a df from parsed XML
    for node in xroot:
        res = []
        for el in df_cols:
            if node is not None and node.find(el) is not None:
                res.append(node.find(el).text)
            else:
                res.append(None)
        rows.append({df_cols[i]: res[i] for i, _ in enumerate(df_cols)})
    # Return the df
    return pd.DataFrame(rows, columns=df_cols)


# Read test data
data = pd.read_csv("./data/test.csv")
data = data.to_json()

# making the api request
r = requests.post(
    url="http://127.0.0.1:3000/predict/", data=data, headers={"User-Agent": "Custom"}
)

df_result = parse_XML(
    r.content,
    df_cols=[
        "Time",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "Amount",
        "Predictions",
    ],
)

# Save results
df_result.to_csv("api_pred_result.csv")
