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
