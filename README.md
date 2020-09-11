
# Named-Entity-Recognition


Pipline Model


数据描述：

数据集是关于医疗诊断的，json格式存储，一个样本，分别包含text和spo_list，spo_list包含一个或多个对象。spo_list里的predicate是要提取的关系，subject是头实体，object是尾实体。

模型架构：

关系抽取的思路是将任务转化成分类任务来做，首先将label抽取出来，然后采用分类模型训练，然后根据输入文本预测对应的关系。采用ALBERT预训练模型+bert4keras框架。

实体识别的思路是采用MRC方式，构建query+passage，来预测start和end位置。框架是ELECTRA预训练模型+bert4keras。

bert4keras用的是0.8.4版本，tensorflow用的是1.15.0版本。

实践结果：

关系分类f1_score可以达到70%，实体识别80%多。

