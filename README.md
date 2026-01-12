<p align="center">
  <img width="800" alt="LLM-0"
       src="https://github.com/user-attachments/assets/0cc855dc-c711-4e44-9e25-5c36df039306"
       style="max-width: 100%;" />

  <br/>
  <img src="https://readme-typing-svg.herokuapp.com?font=Noto+Sans+SC&weight=900&size=28&duration=3000&pause=800&color=000000&background=FFFFFF&center=true&vCenter=true&width=800&height=80&lines=🚀速通RAG、Agent、入门项目、面试八股｜上岸记录🎉"/>
  <br/>

  <a href="https://github.com/limouren2000/llms-dev-study/stargazers">
    <img src="https://img.shields.io/github/stars/limouren2000/llms-dev-study?style=for-the-badge&logo=github&color=brightgreen" />
  </a>
  <a href="https://github.com/limouren2000/llms-dev-study/network/members">
    <img src="https://img.shields.io/github/forks/limouren2000/llms-dev-study?style=for-the-badge&logo=github&color=blue" />
  </a>
  <!-- <a href="https://xhslink.com/m/AteKDs8OWml">
    <img src="https://img.shields.io/badge/xhs-小红书-ff2442?style=for-the-badge&logo=xiaohongshu" />
  </a> -->
</p>


本仓库为本人学习🔥大模型应用开发🔥时整理的核心学习资料，此学习路线主打 **“最快速上岸”**，全是干货无冗余扩展，💰以高效求职搞钱为第一要务💰。

欢迎阅读仓库内容，如果对你有用，麻烦点一下 🌟 star，谢谢！

# ✅ 导读

🚀 本项目为大模型应用开发 RAG 和 Agent 的**学习路线**和**面试八股**，包含最**基础的扫盲课程**，和**系统的优化课程**，主要是协助大家🧐快速入门🧐。  
⚠️ 注意：因为langchain官方的**包版本机制混乱**，这里的一些包大概率是过时了，**解决方案**也很简单：直接把你的报错扔给任何一个大模型（deepseek，GPT，doubao都可以），他们会告诉你怎么解决。  
📖 目录结构为三部分：
- **1.RAG 文件夹**：RAG相关的项目Demo和课程；
- **2.Agent 文件夹**：Agent相关的Demo和课程；
- **3.Interview**：大模型RAG和Agent的面试八股。

# ✅ RAG

本部分一共**四个部分**：
- llms-1和llms-2为B站上的🕶️**扫盲课**🕶️，两位Up主讲的清楚且简洁，主要是入门了解的，快速过一下即可；
- llms-3为Langchain官方出的RAG教程，视频部分这里展示了原版和国内翻译版，主要讲解了RAG过程中的主要流程及其优化点，**🔥建议重点看这个，面试会问很多优化点🔥**；
- llms-4为langchain是官方给出的**💡RAG项目💡**例子，这里会包含最基础的RAG项目的流程，保证你立马就能run起来，并且代码结构很简单。

## llms-1
### 视频地址（看整个系列）：
- https://www.bilibili.com/video/BV1qC4y1F7Dy
### 代码：
- 🌹代码地址（可运行版）：https://github.com/limouren2000/llms-dev-study/tree/main/1.RAG/llms-1/
  - note（✅建议下载运行这个✅）：本人运行代码，部分包的更新（原作者代码部分包过期）
  - original：原作者代码（同代码原址，不建议，可能需要更新包）
- 代码原址：https://github.com/blackinkkkxi/RAG_langchain/tree/main   
- 运行平台：除了langchain_hf，都可以在Colab运行；Kaggle都可以运行，Kaggle入门参考：[白嫖免费算力，量小但管够——Kaggle](https://mp.weixin.qq.com/s/SK5VXzx2zijzjc8OYJICKA)；
## llms-2
### 视频地址（看单篇既可）：
- https://www.bilibili.com/video/BV1Cp421R7Y7
### 代码：
- 🌹代码地址（可运行版）：https://github.com/limouren2000/llms-dev-study/tree/main/1.RAG/llms-2/
  - note（✅建议下载运行这个✅）：本人运行代码，部分包的更新（原作者代码部分包过期）
  - original：原作者代码（同代码原址，不建议，可能需要更新包）
- 代码原址：https://github.com/owenliang/rag-retrieval/tree/main  
- 运行平台：Kaggle运行，Kaggle入门参考：[白嫖免费算力，量小但管够——Kaggle](https://mp.weixin.qq.com/s/SK5VXzx2zijzjc8OYJICKA)；
## llms-3
### 视频地址（看整个系列）：
- 外网原视频（英文）：https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x
- 国内中文版：https://www.bilibili.com/video/BV1dm41127jc/
### 代码：
- 🌹代码地址（可运行版）：https://github.com/limouren2000/llms-dev-study/tree/main/1.RAG/llms-3/
  - note（✅建议下载运行这个✅）：本人运行代码，部分包的更新（原作者代码部分包过期）
  - original：原作者代码（同代码原址，不建议，可能需要更新包）
  - PPT：官方视频对应的PPT
- 代码原址：https://github.com/langchain-ai/rag-from-scratch/tree/main  
- 运行平台：Kaggle运行，Kaggle入门参考：[白嫖免费算力，量小但管够——Kaggle](https://mp.weixin.qq.com/s/SK5VXzx2zijzjc8OYJICKA)；
## llms-4（RAG入门项目）
### 代码：
- 🌹代码地址（✅本人更改过后的代码，可以直接运行✅）：https://github.com/limouren2000/chat-langchain-study/
- 代码原址（运行有问题，需要自己更改）：https://github.com/langchain-ai/chat-langchain
### 说明：
- langchain-chat是官方给出的RAG项目例子，也是我推荐给各位的入门级项目，应网友要求，录制了手把手运行视频，保证你能运行起来。
### 参考资料：
- https://www.bilibili.com/video/BV1eB4y1Z752/
- https://github.com/webup/agi-talks/blob/master/301-langchain-chatdoc/src/slides.md
- https://blog.langchain.dev/building-chat-langchain-2/

# ✅ Agent

本部分一共**三个部分**：
- **1.AI_Agent** 和 **2.QW_Agent** 是B站两个简单的 Agent Demo，比较通俗易懂，主要是入门了解的，快速过一下即可；
- **3.Google_and_Kaggle** 为谷歌联合Kaggle于2025.11.10——2025.11.14推出了他们的实践性课程 ——— AI Agent 强化课程。



## 1.AI_Agent
### 视频地址：
- https://www.bilibili.com/video/BV1JV411F7Yj/
### 代码：
- 🌹代码地址（✅本人更改过后的代码，可以直接运行✅）：https://github.com/limouren2000/llms-dev-study/tree/main/2.Agent/1.AI_Agent
- 代码原址：https://github.com/parallel75/AI_Agent  
- 运行平台：本地

## 2.QW_Agent
### 视频地址：
- https://www.bilibili.com/video/BV1QF4m177Rx/
### 说明：
- 这个项目运行起来需要申请【千问相关key和api】，有坑，建议下载本人更改过后的代码，可以直接运行，千问更新版本需要代码更新，我已更改。
### 代码：
- 🌹代码地址（✅本人更改过后的代码，可以直接运行✅）：https://github.com/limouren2000/llms-dev-study/tree/main/2.Agent/2.QW_Agent
- 代码原址（运行有问题，需要自己更改）：https://github.com/owenliang/agent
- 运行平台：本地

## 3.Google_and_Kaggle

课程介绍官网：https://www.kaggle.com/learn-guide/5-day-agents  
下面是对整个课程的快速解读，可以让你更容易的上手Agent课程。

每天的课程都包含以下三部分：
1. 代码：课程视频中使用的配套代码。
2. 课程视频：官方课程的录播，整体节奏为：课程总览|课程大纲（白皮书）｜Q&A(类似圆桌会议)｜codelabs（课程配套代码解读）｜随堂小测｜总结。
3. 白皮书及其解读：相关技术的白皮书。

### 第1天——Agents介绍
1. 代码：https://github.com/limouren2000/llms-dev-study/tree/main/2.Agent/3.Google_and_Kaggle/1-Day/codelabs
2. 课程视频：https://www.bilibili.com/video/BV1UQm3BPEzX
3. 白皮书及其解读：
- 白皮书：https://github.com/limouren2000/llms-dev-study/blob/main/2.Agent/3.Google_and_Kaggle/1-Day/whitepaper/Introduction%20to%20Agents.pdf
- 播客：https://www.bilibili.com/video/BV12jmwBKEzx

### 第2天——Agent 工具以及与 （MCP） 的互作性
1. 代码：https://github.com/limouren2000/llms-dev-study/tree/main/2.Agent/3.Google_and_Kaggle/2-Day/codelabs
2. 课程视频：https://www.bilibili.com/video/BV1pHm3B9EN6/
3. 白皮书及其解读：
- 白皮书：https://github.com/limouren2000/llms-dev-study/blob/main/2.Agent/3.Google_and_Kaggle/2-Day/whitepaper/Agent%20Tools%20%26%20Interoperability%20with%20Model%20Context%20Protocol%20(MCP).pdf
- 播客：https://www.bilibili.com/video/BV127mwBvENJ/

### 第3天——上下文工程：会话和记忆
1. 代码：https://github.com/limouren2000/llms-dev-study/tree/main/2.Agent/3.Google_and_Kaggle/3-Day/codelabs
2. 课程视频：https://www.bilibili.com/video/BV1E8mRBMEMR/
3. 白皮书及其解读：
- 白皮书：https://github.com/limouren2000/llms-dev-study/blob/main/2.Agent/3.Google_and_Kaggle/3-Day/whitepaper/Context%20Engineering_%20Sessions%20%26%20Memory.pdf
- 播客：https://www.bilibili.com/video/BV1iqmwBQEBn/

### 第4天——Agent 质量
1. 代码：https://github.com/limouren2000/llms-dev-study/tree/main/2.Agent/3.Google_and_Kaggle/4-Day/codelabs
2. 课程视频：https://www.bilibili.com/video/BV1PBmYBpE8X/
3. 白皮书及其解读：
- 白皮书：https://github.com/limouren2000/llms-dev-study/blob/main/2.Agent/3.Google_and_Kaggle/4-Day/whitepaper/Agent%20Quality.pdf
- 播客：https://www.bilibili.com/video/BV1iqmwBQEtH/?

### 第5天——原型到生产

1. 代码：https://github.com/limouren2000/llms-dev-study/tree/main/2.Agent/3.Google_and_Kaggle/5-Day/codelabs
2. 课程视频：https://www.bilibili.com/video/BV1ArmYBnEJW/
3. 白皮书及其解读：
- 白皮书：https://github.com/limouren2000/llms-dev-study/blob/main/2.Agent/3.Google_and_Kaggle/5-Day/whitepaper/Prototype%20to%20Production.pdf
- 解读：https://www.bilibili.com/video/BV1qimwBpEqd/

## 4.Agent入门项目
 - 这里是Agent的入门项目，使用 MCP 快速搭建一个 Agent：https://github.com/limouren2000/easy-mcp

# ✅ Interview
本部分包含两部分面试八股，是本人在找工作期间收集和整理的大模型应用开发八股文，本人实测，可以通过这些找到了一些大厂**高级AI研发工程师**相关岗位。
- [大模型应用开发八股](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=Mzk1NzgzMjY3OQ==&action=getalbum&album_id=3987723560113356813&scene=126&uin=&key=&devicetype=iMac+MacBookPro18%2C3+OSX+OSX+15.4.1+build(24E263)&version=13080a10&lang=zh_CN&nettype=WIFI&ascene=78&fontScale=100)
## RAG
- 详细介绍大模型（LLMs）RAG检索增强生成学习/面试过程中可能遇到的知识点，全文4w+字，按照处理流程整理：[大模型RAG知识笔记](https://mp.weixin.qq.com/s/zmUTGAMoljXSmnoo_cBQig)
## Agent
- 详细介绍大模型（LLMs）智能体Agent学习/面试过程中可能遇到的知识点，全文1w+字，按照模块整理：[大模型Agent知识笔记](https://mp.weixin.qq.com/s/TSioLS_RhrX57YEnY3mkag)

# ✅ 增长曲线

<p align="center">
  <!-- Star History 增长曲线 -->
  <img width="800" alt="Star History Chart"
       src="https://api.star-history.com/svg?repos=limouren2000/llms-dev-study&type=Date"
       style="max-width: 100%;" />
</p>

# ✅ 说明

本仓库的内容**足够支撑基础学习和面试准备**，但收到不少朋友反馈：希望获得**更细致的学习规划**、**项目实操指导**，或是遇到问题能及时得到**针对性答疑**。

由于个人精力有限，一对一答疑、定制化学习路线梳理、项目细节拆解等服务需要**占用大量私人时间**，因此在小红书上架了更完整的配套服务（包含**不同基础的详细学习路线**、完整版**面试八股**、**项目包装**攻略、**专属答疑**通道）。

如果需要更深度的指导，帮你**少走弯路**、高效**突破学习瓶颈**，可点击下方徽章了解详情，我会尽力为大家解决实际问题～

<p align="center">
  <a href="https://xhslink.com/m/AnSrQOLt5y1" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/xhs-小红书配套服务-ff2442?style=for-the-badge&logo=xiaohongshu" />
  </a>
</p>

