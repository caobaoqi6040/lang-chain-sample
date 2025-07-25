"""
多服务器 MCP + LangChain Agent 示例
---------------------------------
1. 读取 .env 中的 LLM_API_KEY / BASE_URL / MODEL
2. 读取 servers_config.json 中的 MCP 服务器信息
3. 启动 MCP 服务器（支持多个）
4. 将所有工具注入 LangGraph Agent，由大模型自动选择并调用
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

# 设置记忆存储
checkpointer = InMemorySaver()

# 读取提示词
prompt = """
            你是一个智能体,具备以下两种能力:
                1.查询天气: 调用 query_weather(city: str)，返回指定城市的实时天气
                2.写入文件: 调用 write_file(content: str)，将文本内容写入本地文件并返回路径
                
            当用户提出请求时，你需要理解意图并选择相应的工具.如果请求缺少必要信息,先与用户确认后再调用工具.返回结果时以简洁、友好的方式回复.
            如果用户提出的问题与你的功能无关,请礼貌告知无法处理.
        """

# 设置对话配置
config = {
    "configurable": {
        "thread_id": "1"
    }
}


# ────────────────────────────
# 环境配置
# ────────────────────────────

class Configuration:
    """读取 .env 与 servers_config.json"""
    def __init__(self) -> None:
        load_dotenv()
        self.api_key: str = os.getenv("DEEPSEEK_API_KEY") or ""
        self.base_url: str | None = os.getenv("DEEPSEEK_BASE_URL")  # DeepSeek 用 https://api.deepseek.com
        self.model: str = os.getenv("MODEL") or "deepseek-chat"
        if not self.api_key:
            raise ValueError("❌ 未找到 LLM_API_KEY，请在 .env 中配置")

    @staticmethod
    def load_servers(file_path: str = "servers_config.json") -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f).get("mcpServers", {})


# ────────────────────────────
# 主逻辑
# ────────────────────────────
async def run_chat_loop() -> None:
    """启动 MCP-Agent 聊天循环"""
    cfg = Configuration()
    if cfg.base_url:
        os.environ["DEEPSEEK_API_BASE"] = cfg.base_url
    servers_cfg = Configuration.load_servers()
    # 1️ 连接多台 MCP 服务器
    mcp_client = MultiServerMCPClient(servers_cfg)

    tools = await mcp_client.get_tools()  # LangChain Tool 对象列表

    logging.info(f"✅ 已加载 {len(tools)} 个 MCP 工具： {[t.name for t in tools]}")

    model = ChatDeepSeek(model="deepseek-chat")

    # 3 构造 LangGraph Agent
    agent = create_react_agent(model=model,
                               tools=tools,
                               prompt=prompt,
                               checkpointer=checkpointer)

    # 4 CLI 聊天
    print("\n🤖 MCP Agent 已启动，输入 'quit' 退出")
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() == "quit":
            break
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config
            )
            print(f"\nAI: {result['messages'][-1].content}")
        except Exception as exc:
            print(f"\n⚠️  出错: {exc}")

    # 5️ 清理
    await mcp_client.cleanup()
    print("🧹 资源已清理，Bye!")


# ────────────────────────────
# 入口
# ────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    asyncio.run(run_chat_loop())
