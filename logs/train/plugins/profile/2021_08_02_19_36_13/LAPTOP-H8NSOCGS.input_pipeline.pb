  *	???a??@2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map??.???@!	??X?D@)V??? >@1׵i|??C@:Preprocessing2U
Iterator::Model::ParallelMapV2?ip[?6@!?8????=@)?ip[?6@1?8????=@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch[B>??\1@!????S?6@)[B>??\1@1????S?6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap	?I*S?4@!S??!	?:@)?S^m@1Av?4!@:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeata??????!??v@)G?I?Q??1.??K?M@:Preprocessing2F
Iterator::Modelo???TY7@!y?^ P?>@)?8?#+???1B???????:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat	y?|???!?{'?QN??)-Z??լ??1>?????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	? ??4@!u]{N}?;@)bMeQ?E??17?o?r???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	?x?????!?????J??)?x?????1?????J??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor^J]2????!.??n??)^J]2????1.??n??:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate	q̲'?͹?!0uRR????)???+?,??1h??G?D??:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range?pA?,??!???`?~??)?pA?,??1???`?~??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.