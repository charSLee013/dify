# 介绍
基于原版 [orginal repo](https://github.com/langgenius/dify)进行一些魔改，对原来的逻辑(包括部署)有一些区别，可以称为是 **nightly** 版本

# 已知修改的地方
- [x] `docker-compose`默认`vector database`改为`milvus`并内置
- [x] 添加一个环境变量`QA_MODEL_CONCURRENCY`用来控制每次QA文档分割后发送给LLM的并发数量。
  原来逻辑是10个线程并发请求，如果选用的模型是CPU运行或者模型供应商有API请求限制(比如智谱 AI)会导致**超时**或者**API访问受限**等问题

# 目前计划

- [] 支持自定义的OpenAI的**API服务**
- [] 支持**文件夹上传**和新的上传文件类型检查(不受限于单纯的文件后缀名)
- [] 提高QA分割时候的最大分段数量
  

# 路线图

## QA分割需要更多针对性改进
一般情况下文档分段设置为384～512之间为embedding模型所容乃的范围内最好，但是QA分割是将文档发送给LLM后得出N个问答对，可LLM目前Token上限已经在8K~32K之间了，大大超过了Embedding模型的范围
所以对于QA分割最好的方式是：
  1. 设定最大分段大小为LLM最大长度的70%
  2. 对于pdf,word这些可以按**页**先进行分割，然后再遵循QA分割规则


## 上传文件
现在上传文件限定在 txt,html,md,pdf,xlsx,csv,docx 类型，没有考虑一些纯文本的文件(比如编程语言文件)
需要设立白名单和黑名单规则，如果不在黑白名单内的话，则读取文件的前64字节是否为可读的ASCII或者UTF-8字符，以此来判断是否为文本文件

再新增一个上传文件夹的功能，该功能能上传目标文件夹以及子文件夹内全部符合规则的文件,并且将文件名称设定为上传文件夹的相对路径
比如说 `/app/api/core/model.py` 文件，将`app`文件夹上传之后，文件名称为 `api/core/model.py`，这样就能区分相同名称的不同路径的文件(windows 下还会有同名文件和文件夹的情况)

## 多模态支持
不仅是chatGPT4的多模态，而是利用多模态模型跟LLM模型一起工作
比如 [图像<=>文本模型BLIP](https://github.com/salesforce/BLIP)
这样我们可以在对话中上传多媒体文件(知识库也可以考虑将多模态纳入，但保存多媒体文件是一件很麻烦的事情)

