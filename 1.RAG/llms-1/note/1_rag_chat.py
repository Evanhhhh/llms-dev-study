"""
基于HuggingFace的RAG对话系统 - 无需API key
"""

# ============================================
# 第一部分: 安装依赖
# ============================================
"""
运行以下命令安装依赖:
pip install langchain sentence-transformers chromadb pypdf transformers torch
"""

# ============================================
# 第二部分: 导入必要的库
# ============================================
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 使用HuggingFace的embedding模型
from langchain.embeddings import HuggingFaceEmbeddings

# 新增: 导入Ollama
from langchain.llms import Ollama

print("✅ 库导入成功!")
# ============================================
# 第三部分: 加载并处理文档
# ============================================

# 1. 加载PDF文档 (替换为你的文档路径)
print("\n📄 加载文档...")
loader = PyPDFLoader("https://arxiv.org/pdf/2309.10305.pdf")  #关于 "Baichuan2" 模型的论文 PDF
data = loader.load()

print(f"✅ 加载了 {len(data)} 个文档")

# 2. 文本分割
print("\n✂️ 分割文档...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每块的字符数
    chunk_overlap=50,      # 块之间的重叠
    length_function=len,
)

docs = text_splitter.split_documents(data)
print(f"✅ 分割为 {len(docs)} 个文本块")

# ============================================
# 第四部分: 创建向量嵌入(使用HuggingFace)
# ============================================

print("\n🔢 创建向量嵌入模型...")

# 使用HuggingFace的开源embedding模型(免费,无需API key)
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # 这是一个支持中文的轻量级模型
    model_kwargs={'device': 'cuda'},  # 使用GPU,如果用CPU可改为'cpu'
    encode_kwargs={'normalize_embeddings': True}
)

print("✅ Embedding模型加载成功!")

# ============================================
# 第五部分: 构建向量数据库
# ============================================

print("\n💾 构建向量数据库...")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embed_model,  # 使用HuggingFace的embedding
    collection_name="hf_embed",
    persist_directory="./chroma_db"  # 数据持久化目录
)

print("✅ 向量数据库创建成功!")

# 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 3}  # 返回最相关的3个文档块
)

# ============================================
# 第六部分: 加载Ollama语言模型
# ============================================

print("\n🤖 加载语言模型...")

# 方案1: 使用Ollama(推荐!简单快速)
try:
    llm = Ollama(
        model="qwen:1.8b",      # 使用Qwen 1.8B中文模型
        # 其他可选模型:
        # model="llama2:7b"     # 英文模型
        # model="qwen:7b"       # 更大的中文模型
        temperature=0.7,        # 控制创造性(0-1)
    )
    
    # 测试Ollama是否正常工作
    test_response = llm("你好")
    print("✅ Ollama语言模型加载成功!")
    print(f"   测试回复: {test_response[:50]}...")
    
except Exception as e:
    print(f"⚠️ Ollama加载失败: {e}")
    print("💡 请确保:")
    print("   1. 已安装Ollama (访问 https://ollama.ai)")
    print("   2. 已下载模型 (运行: ollama pull qwen:1.8b)")
    print("   3. Ollama服务正在运行")
    print("\n💡 将使用仅检索模式...")
    llm = None


# ============================================
# 第七部分: 创建RAG问答链
# ============================================

print("\n🔗 创建问答链...")

# 定义提示模板
template = """请基于以下上下文回答问题。如果无法从上下文中找到答案,请说"我不知道"。

上下文: {context}

问题: {question}

回答:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

if llm:
    # 创建完整的问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    print("✅ 问答链创建成功!")
else:
    print("⚠️ 仅使用检索功能演示")

# ============================================
# 第八部分: 测试问答系统
# ============================================

def ask_question(question):
    """提问函数"""
    print(f"\n❓ 问题: {question}")
    print("-" * 60)
    
    if llm:
        # ⭐⭐⭐ 使用完整的RAG系统(有Ollama) ⭐⭐⭐
        result = qa_chain({"query": question})
        
        # 显示AI生成的回答
        print(f"\n💡 AI回答:\n{result['result']}")
        
        # 显示来源文档
        print(f"\n📚 参考来源:")
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"{i}. {doc.page_content[:100]}...")
    else:
        # 仅演示检索功能(没有Ollama)
        docs = retriever.get_relevant_documents(question)
        print("📚 检索到的相关文档:")
        for i, doc in enumerate(docs, 1):
            print(f"\n{i}. {doc.page_content}")

# ============================================
# 第九部分: 运行示例
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 RAG对话系统已就绪!")
    print("=" * 60)
    
    # 测试问题
    questions = [
        "How large is the baichuan2 vocabulary?",
        "你知道baichuan2模型吗？",
        "向量数据库的作用是什么?"
    ]
    
    for q in questions:
        ask_question(q)
        print("\n")
    
    # 交互式问答
    print("\n💬 现在可以开始提问了!(输入'退出'结束)")
    while True:
        question = input("\n你的问题: ").strip()
        if question.lower() in ['退出', 'exit', 'quit', 'q']:
            print("👋 再见!")
            break
        if question:
            ask_question(question)