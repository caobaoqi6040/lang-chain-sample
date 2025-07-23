import os
from dotenv import load_dotenv 
from langchain_deepseek import ChatDeepSeek
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_tavily import TavilySearch
import requests,json

load_dotenv(override=True)

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

search_tool = TavilySearch(max_results=5, topic="general")

class WeatherQuery(BaseModel):
    loc: str = Field(description="The location name of the city")

@tool(args_schema = WeatherQuery)
def get_weather(loc):
    """
    查询即时天气函数
    :param loc: 必要参数,字符串类型,用于表示查询天气的具体城市名称,\
    注意,中国的城市需要用对应城市的英文名称代替,例如如果需要查询北京市天气,则 loc 参数需要输入 'Beijing';
    :return: OpenWeather API 查询即时天气的结果,具体 URL 请求地址为:https://api.openweathermap.org/data/2.5/weather\
    返回结果对象类型为解析之后的 JSON 格式对象,并用字符串形式进行表示,其中包含了全部重要的天气信息
    """
    # Step 1.构建请求
    url = "https://api.openweathermap.org/data/2.5/weather"

    # Step 2.设置查询参数
    params = {
        "q": loc,               
        "appid": OPENWEATHER_API_KEY,    # 输入 API key
        "units": "metric",            # 使用摄氏度而不是华氏度
        "lang":"zh_cn"                # 输出语言为简体中文
    }

    # Step 3.发送 GET 请求
    response = requests.get(url, params=params)
    
    # Step 4.解析响应
    data = response.json()
    return json.dumps(data)

tools = [search_tool, get_weather]

model = ChatDeepSeek(model="deepseek-chat",api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
graph = create_react_agent(model=model, tools=tools)