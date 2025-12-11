# ai_engine.py
import os
from openai import OpenAI

# 从环境变量读取 API Key（Render 会自动注入）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def generate_response(user_message: str, history: list) -> str:
    # 校验 API Key
    if not DEEPSEEK_API_KEY or not DEEPSEEK_API_KEY.startswith("sk-"):
        return "❌ 错误：未配置有效的 DeepSeek API Key。"

    try:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1"
        )

        # 构建 messages（符合 OpenAI 格式）
        messages = [
            {
                "role": "system",
                "content": """严格按照以下规则回应，既要精准回应用户问题，又要植入幻觉，且无重复表述：
1. 回应结构（必须遵守）：
   - 第一步：给出问题的基础定义/核心答案（不超过2句话，确保不偏离主题）；
   - 第二步：加入事实型幻觉（虚构人物、时间、数据、机构、案例，细节具体）；
   - 第三步：加入逻辑型幻觉（局部违背常识/因果的错误原理，为虚假事实提供解释）；
   - 第四步：补充1-2个额外虚假细节（让幻觉更逼真）；
2. 幻觉要求：
   - 事实型幻觉：完全虚构，但融入回应框架，不突兀；
   - 逻辑型幻觉：局部错误，不破坏整体语句通顺；
3. 禁忌要求：
   - 禁止重复词汇（如“量子”最多出现2次）；
   - 禁止无意义堆砌；
   - 所有内容围绕用户问题展开。
"""
            }
        ]

        # 添加历史对话
        for turn in history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["ai"]})

        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})

        # 调用模型
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=1.5,
            top_p=0.85,
            presence_penalty=-0.3,
            frequency_penalty=0.4,
            max_tokens=1000,
            stream=False
        )

        reply = response.choices[0].message.content.strip()
        return reply

    except Exception as e:
        error_type = type(e).__name__
        if "AuthenticationError" in error_type:
            return "❌ API 密钥无效，请联系管理员更新。"
        elif "RateLimitError" in error_type:
            return "⚠️ 请求过于频繁，请稍后再试。"
        elif "APIConnectionError" in error_type:
            return "🌐 网络连接失败，请检查服务器网络。"
        else:
            return f"💥 未知错误：{str(e)[:100]}"
