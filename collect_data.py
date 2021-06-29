from urllib.request import urlopen, Request
from urllib.parse import urlencode, quote_plus
import xmltodict
import pandas as pd
import numpy as np
import os

# KCDA의 openapi 주소와 key
service_url = 'http://openapi.data.go.kr/openapi/service/rest/Covid19/getCovid19InfStateJson'
service_key = '' # API 신청 후 받은 키 입력

start_date = '20200301' # 요청 시작일
end_date = '20210626' # 요청 마감일
page_no = '1' # 페이지 번호
num_of_rows = '10' # 페이지당 결과 수

fname = f'covid_{start_date}_{end_date}.csv'


def get_response():
    '''
    Args:
    Returns:
        response_body (str): xml형태의 문자열
        
    xml형태는 https://www.data.go.kr/data/15043376/openapi.do 를 참고
    '''
    query_params = '?' + urlencode({
        quote_plus('ServiceKey'): service_key,
        quote_plus('pageNo'): page_no,
        quote_plus('numOfRows'): num_of_rows,
        quote_plus('startCreateDt'): start_date,
        quote_plus('endCreateDt'): end_date})
    
    req = Request(service_url+query_params)
    req.get_method = lambda: 'GET'
    response_body = urlopen(req).read()
    return response_body


def convert_response(response):
    '''
    Args:
        response (str): xml형태의 문자열

    Returns:
        None

        xml형태의 문자열을 csv형태로 변환하여 저장
    '''
    
    response = xmltodict.parse(response)['response']
    covid_data = pd.DataFrame(response['body']['items']['item'])
    covid_data = covid_data.astype({
        'accDefRate': 'float32',
        'accExamCnt': 'int32',
        'accExamCompCnt': 'int32',
        'careCnt': 'int32',
        'clearCnt': 'int32',
        'createDt': 'datetime64[ns]',
        'updateDt': 'datetime64[ns]',
        'deathCnt': 'int32',
        'decideCnt': 'int32',
        'examCnt': 'int32',
        'resutlNegCnt': 'int32',
        'seq': 'int32',
    }, errors='ignore')

    # datetime 타입의 컬럼은 값이 없을 경우 'null' 문자열이 들어있음
    # 해당 값을 None으로 변경
    object_columns = covid_data.dtypes[covid_data.dtypes=='object'].index
    for col in object_columns:
        covid_data[col] = covid_data[col].apply(lambda x: x if x != 'null' else None)

    covid_data = covid_data.sort_values(by=['stateDt'], ignore_index=True)
    covid_data['decideDailyCnt'] = [np.nan] + [covid_data['decideCnt'][i]-covid_data['decideCnt'][i-1] 
                                                for i in range(1, len(covid_data))]
    covid_data = covid_data.dropna(subset=['decideDailyCnt'])
    covid_data = covid_data.astype({'decideDailyCnt': 'int32'}, errors='ignore')
    os.makedirs('./data', exist_ok=True)
    covid_data.to_csv(f'./data/{fname}', index=False)


if __name__=='__main__':
    print(f'Request to "{service_url}"')
    response = get_response()
    print('convert response into csv file')
    convert_response(response)
    print('done')







