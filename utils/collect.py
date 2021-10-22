from urllib.request import urlopen, Request
from urllib.parse import urlencode, quote_plus
from sklearn.preprocessing import StandardScaler
import xmltodict
import pickle
import pandas as pd
import numpy as np
import os


with open("./kcda_api_key.txt", "r") as f:
    service_key = f.read() # API 키를 파일로 부터 읽음

start_date = "20200303" # 요청 시작일
end_date = "20210805" # 요청 마감일
page_no = "1" # 페이지 번호
num_of_rows = "10" # 페이지당 결과 수

data_path = "./data"
fname = f"covid_{start_date}_{end_date}.csv"


def get_nation_covid_data(start_date, end_date, page_no=page_no, num_of_rows=num_of_rows):
    """ 공공데이터포털 API를 사용해 국내 코로나 현황을 불러오고 저장합니다.
    
    데이터에 대한 자세한 내용은 "https://www.data.go.kr/tcs/dss/selectApiDataDetailView.do?publicDataPk=15043376"를 확인하세요.

    Args:
        state_date (str): 요청 시작일 ex) "20200301"
        end_date (str): 요청 마감일 ex) "20200601"
        page_no (str): 페이지 번호 ex) "1"
        num_of_rows (str):  페이지당 결과 수 ex) "10"

    Returns:
        covid_data (pandas.DataFrame): 데이터를 테이블 형태(.csv)로 저장

    Raises:
        ValueError: API 요청에 대한 결과가 정상적이지 않을 때 발생
    """

    # 파일 이름
    fname = f"covid_{start_date}_{end_date}.csv"
    # 국내 코로나 현황 API
    service_url = "http://openapi.data.go.kr/openapi/service/rest/Covid19/getCovid19InfStateJson"
    query_params = "?" + urlencode({
        quote_plus("ServiceKey"): service_key,
        quote_plus("pageNo"): page_no,
        quote_plus("numOfRows"): num_of_rows,
        quote_plus("startCreateDt"): start_date,
        quote_plus("endCreateDt"): end_date})
    
    req = Request(service_url+query_params)
    req.get_method = lambda: "GET"
    response = urlopen(req)

    if response.status == 200:
        response = xmltodict.parse(response.read())["response"]
        if response["header"]["resultCode"] == "00":
            if response["body"]["items"]:
                covid_data = pd.DataFrame(response["body"]["items"]["item"])
                covid_data = covid_data.replace({"null": None, "NULL": None, "-": None})
                covid_data = covid_data.sort_values(by=["stateDt"], ignore_index=True)
                os.makedirs(data_path, exist_ok=True)
                scaler = StandardScaler()
                scaler.fit(np.diff(covid_data[["decideCnt"]].astype("int32"), n=2, axis=0))
                with open(f"{data_path}/{fname[:-4]}_scaler.pkl", "wb") as f:
                    pickle.dump(scaler, f)

                covid_data.to_csv(f"{data_path}/{fname}", index=False)
                return covid_data
    raise ValueError(f"could not get response value from API, {service_url}, {response}")


if __name__ == "__main__":
    print(f"Request to API and save data from {start_date} to {end_date}")
    get_nation_covid_data(start_date, end_date)
    print("Done")
