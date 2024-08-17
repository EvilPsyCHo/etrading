# API_URL = "http://89z06660x2.zicp.fun"

import requests
import json
from config import URL


class DataAPI:
    # url = "http://89z06660x2.zicp.fun"
    url = URL

    # def __init__(self, url=None) -> None:
    #     self.url = url or API_URL

    def test(self):
        suffix = "/chengDuController/queryGnEngineeringByCondition"
        response = requests.post(self.url+suffix, json={})
        return response

    def create_eng(self, name, remark=None):
        # 创建工程
        suffix = "/chengDuController/insertGnEngineering"
        json_data = {
        "engName": name,
        "remark": remark}
        response = json.loads(requests.post(self.url + suffix, json=json_data).content)
        return response
    
    def get_eng_info(self):
        # 获取工程列表及工程信息
        suffix = "/chengDuController/queryGnEngineeringByCondition"
        response = requests.post(self.url+suffix, json={})
        response = json.loads(response.content)
        return response

    def load_eng_base_data(self, eng_id):
        # 工程加载基础数据
        suffix = "/chengDuController/loadingData"
        params = {"engId": eng_id}
        response = json.loads(requests.post(self.url+suffix, params=params).content)
        return response
    
    def load_eng_typical_data(self, eng_id, typical_id):
        # 工程加载典型数据
        suffix = "/chengDuController/loadingTypicalData"
        params = {"engId": eng_id, "oldEngId": typical_id}
        response = json.loads(requests.post(self.url+suffix, params=params).content)
        return response
    
    def get_eng_bus_load(self, eng_id):
        # 获取工程母线负荷
        suffix = "/chengDuController/queryEngLoadByCondition"
        params = {"engId": eng_id}
        response = json.loads(requests.post(self.url+suffix, params=params).content)
        return response

    def get_eng_system_load(self, eng_id):
        # 获取工程系统负荷
        suffix = "/chengDuController/queryEngSystemLoadByCondition"
        params = {"engId": eng_id}
        response = json.loads(requests.post(self.url+suffix, params=params).content)
        return response

    def get_eng_new_energy_forecast(self, eng_id):
        # 获取新能源负荷预测
        suffix = "/chengDuController/queryEngNewenergyPlanByCondition"
        response = json.loads(requests.post(self.url+suffix, data={"engId": eng_id}).content)
        return response

    def get_eng_unit_info(self, eng_id):
        # 查询机组数据
        suffix = "/chengDuController/queryEngUnitInfomationByCondition"
        response = json.loads(requests.post(self.url+suffix, json={"engId": eng_id}).content)
        return response
    
    def get_eng_line_map(self, eng_id):
        # 查询母线机组映射表
        suffix = "/chengDuController/queryLineForChengdu"
        response = json.loads(requests.post(self.url+suffix, data={"engId": eng_id}).content)
        return response
    
    def get_eng_unit_contrct(self, eng_id):
        # 查询机组中长期合约
        suffix = "/chengDuController/queryContractCurve"
        data = {"engId": eng_id}
        response = json.loads(requests.post(self.url+suffix, data=data).content)
        return response
    
    def create_eng_declaration(self, data):
        # 提交申报数据
        suffix = "/chengDuController/insertEngUnitDeclare"
        response = json.loads(requests.post(self.url+suffix, json=data).content)
        return response
    
    def create_eng_clearing(self, eng_id):
        # 交易出清计算
        suffix = "/chengDuController/calcuClering"
        data = {"engId": eng_id}
        response = json.loads(requests.post(self.url+suffix, data=data).content)
        return response
    
    def get_eng_clearing_power_and_price(self, eng_id):
        # 查询出清结果，电量与电价
        suffix = "/chengDuController/queryPowerAndPrice"
        data = {"engId": eng_id}
        response = json.loads(requests.post(self.url+suffix, data=data).content)
        return response
    
    def get_eng_clearing_start_and_stop(self, eng_id):
        # 查询出清起停结果
        suffix = "/chengDuController/queryGnEngStartstop"
        data = {"engId": eng_id}
        response = json.loads(requests.post(self.url+suffix, data=data).content)
        return response


if __name__ == "__main__":
    eng_id = "42790604-138d-4386-bd79-156703e7d40a"
    api = DataAPI()
    print(api.get_eng_info())
    # print(api.get_eng_info()["status"])
    # print(api.get_eng_bus_load(eng_id)["status"])
    # print(api.get_eng_unit_info(eng_id)["status"])
    # print(api.get_eng_line_map(eng_id)["status"])
    # print(api.get_eng_clearing_start_and_stop(eng_id)["status"])
