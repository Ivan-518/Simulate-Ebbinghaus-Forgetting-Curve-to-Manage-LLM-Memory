# Simulate Ebbinghaus Forgetting Curve to Manage LLM Memory

> **基于艾宾浩斯遗忘曲线的 LLM 动态记忆管理系统**
>
> *给 AI 装上“类人海马体”：让记忆像人类一样自然衰减与重塑。*


## 项目背景 (Background)

在大语言模型（LLM）的实际应用中，尤其是 **AI 辅助编程 (AI Coding)**、**长篇创作** 或 **长期伴侣** 场景下，面临着核心矛盾：

1.  **非结构化流失**：用户的提问和模型的回答通常是碎片化的，如果不加以整理，关键信息会淹没在“废话”中。
2.  **上下文遗忘 (Context Loss)**：随着对话轮数增加，早期的关键设定（如变量定义、伏笔、用户偏好）被模型遗忘。
3.  **Token 效率低下**：为了保持记忆，传统做法是全量携带历史记录，导致 Token 消耗呈指数级增长，且引入大量噪声。

本项目提出了一种**仿生学**解决方案：**先进行“关键信息提取”，再利用“艾宾浩斯遗忘曲线”对这些信息进行全生命周期管理。**

## 核心机制 (Core Mechanism)

模拟了人类大脑的记忆处理流程，分为三个阶段：

### 1. 自动总结与编码 (Encoding)
无论是用户的输入，还是 AI 的输出，系统都不会直接存储原始文本。
* **动作**：后台调用 LLM 对当前交互进行“提炼”。
* **产物**：生成一条结构化的记忆条目（Memory Item），并由 LLM 赋予其**初始重要性 (Importance)**。
* *示例*：从一段关于“数据校验报错”的 Debug 对话中，提炼出 UserSchema Pydantic 模型中的 age 字段类型从 int 修改为 Optional[int]，且增加了默认值 None，重要性打分 0.85。

### 2. 艾宾浩斯动态存储 (Storage & Decay)
记忆不是静态的，它遵循“用进废退”的自然法则。系统根据时间和频率两个维度动态调整每条记忆的权重。
* **工作记忆 (Working Memory)**：保存在 `context.md` 中，随 Prompt 发送给 LLM。
* **长期归档 (Long-term Archive)**：存储在数据库中，处于沉睡状态，不消耗 Token。

### 3. 唤醒与重塑 (Retrieval & Reconsolidation)
当新的任务触发了旧的记忆（通过语义检索），该记忆会被“唤醒”回到工作记忆区，同时其**复习频率**增加，导致其在未来更难被遗忘（记忆加固）。

## 算法逻辑 (The Algorithm)

系统为每一条记忆计算一个实时的**留存分数 (Retention Score)**。

> **Score = (I × F) - (T × R)**

* **I (Importance - 初始重要性)**：(0.0 - 1.0)。由 LLM 在总结阶段自动打分。
    * *高分*：核心架构、剧情主线、用户忌讳。
    * *低分*：临时变量、寒暄、已修复的报错。
* **F (Frequency - 复习频率)**：该条目被检索/命中的次数。每次被唤醒，F 值增加，显著抵消时间的侵蚀。
* **T (Time - 衰减时间)**：距离上一次被访问（Last Accessed）的时间间隔。
* **R (Rate - 遗忘速率)**：控制遗忘速度的常数系数。

**生命周期判定：**
* 当 `Score > 阈值`：保留在工作记忆（Context）。
* 当 `Score < 阈值`：移入长期归档（Archive）。

## 伪代码 (Pseudo-code)


```python
class EbbinghausMemorySystem:
    def __init__(self):
        # 1. 工作记忆 (Context): 活跃内存，直接发给 LLM
        self.working_memory = [] 
        # 2. 长期归档 (Archive): 沉睡内存，存储在向量库中
        self.long_term_archive = []
        
        # 核心参数
        self.DECAY_RATE = 0.1       # R: 遗忘速率
        self.SCORE_THRESHOLD = 0.5  # 阈值: 低于此分数的记忆将被移出

    # [输入端]：并非直接存储原始对话，而是存储“摘要”
    def add_event(self, raw_text):
        # Step 1: 调用 LLM 进行自动总结与重要性打分
        # 示例: summary="将 requests 库替换为 httpx", importance=0.9
        summary, importance = LLM.analyze(raw_text)
        
        new_memory = {
            "content": summary,
            "importance": importance, # I: 初始重要性
            "frequency": 1,           # F: 复习频率 (初始为1)
            "last_accessed": NOW()    # T: 最后访问时间
        }
        self.working_memory.append(new_memory)

    # [输出端]：构建 Context 前的“唤醒”与“清理”
    def get_context(self, user_query):
        # Step A: 唤醒 (Recall) - 尝试从归档中捞回相关记忆
        # 如果 query 包含 "网络库"，可能会命中归档里的 "httpx" 条目
        related_items = VectorDB.search(self.long_term_archive, user_query)
        for item in related_items:
            item["frequency"] += 1      # F + 1 (复习加固)
            item["last_accessed"] = NOW() # T 重置
            self.working_memory.append(item) # 移回工作区

        # Step B: 维护 (Decay) - 执行艾宾浩斯遗忘算法
        final_context = []
        for mem in self.working_memory:
            # 计算未访问天数
            t_days = NOW() - mem["last_accessed"]
            
            # Score = (重要性 x 频率) - (时间流逝 x 衰减率)
            score = (mem["importance"] * mem["frequency"]) - (t_days * self.DECAY_RATE)
            
            if score > self.SCORE_THRESHOLD:
                final_context.append(mem["content"]) # 保留
            else:
                self.long_term_archive.append(mem)   # 遗忘 (移入归档)
        
        return "\n".join(final_context)
