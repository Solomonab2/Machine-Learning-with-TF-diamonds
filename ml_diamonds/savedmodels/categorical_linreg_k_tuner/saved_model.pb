??
?!?!
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
?
SparseFillEmptyRows
indices	
values"T
dense_shape	
default_value"T
output_indices	
output_values"T
empty_row_indicator

reverse_index_map	"	
Ttype
h
SparseReshape
input_indices	
input_shape	
	new_shape	
output_indices	
output_shape	
?
SparseSegmentMean	
data"T
indices"Tidx
segment_ids"Tsegmentids
output"T"
Ttype:
2"
Tidxtype0:
2	"
Tsegmentidstype0:
2	
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??
?
?sequential_1/dense_features/clarity_embedding/embedding_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*P
shared_nameA?sequential_1/dense_features/clarity_embedding/embedding_weights
?
Ssequential_1/dense_features/clarity_embedding/embedding_weights/Read/ReadVariableOpReadVariableOp?sequential_1/dense_features/clarity_embedding/embedding_weights*
_output_shapes

:*
dtype0
?
=sequential_1/dense_features/color_embedding/embedding_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*N
shared_name?=sequential_1/dense_features/color_embedding/embedding_weights
?
Qsequential_1/dense_features/color_embedding/embedding_weights/Read/ReadVariableOpReadVariableOp=sequential_1/dense_features/color_embedding/embedding_weights*
_output_shapes

:*
dtype0
?
sequential_2/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_namesequential_2/dense_4/kernel
?
/sequential_2/dense_4/kernel/Read/ReadVariableOpReadVariableOpsequential_2/dense_4/kernel*
_output_shapes
:	?*
dtype0
?
sequential_2/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namesequential_2/dense_4/bias
?
-sequential_2/dense_4/bias/Read/ReadVariableOpReadVariableOpsequential_2/dense_4/bias*
_output_shapes	
:?*
dtype0
?
sequential_2/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_namesequential_2/dense_5/kernel
?
/sequential_2/dense_5/kernel/Read/ReadVariableOpReadVariableOpsequential_2/dense_5/kernel*
_output_shapes
:	?*
dtype0
?
sequential_2/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namesequential_2/dense_5/bias
?
-sequential_2/dense_5/bias/Read/ReadVariableOpReadVariableOpsequential_2/dense_5/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
k

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name466*
value_dtype0	
m
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name558*
value_dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
FAdam/sequential_1/dense_features/clarity_embedding/embedding_weights/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*W
shared_nameHFAdam/sequential_1/dense_features/clarity_embedding/embedding_weights/m
?
ZAdam/sequential_1/dense_features/clarity_embedding/embedding_weights/m/Read/ReadVariableOpReadVariableOpFAdam/sequential_1/dense_features/clarity_embedding/embedding_weights/m*
_output_shapes

:*
dtype0
?
DAdam/sequential_1/dense_features/color_embedding/embedding_weights/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*U
shared_nameFDAdam/sequential_1/dense_features/color_embedding/embedding_weights/m
?
XAdam/sequential_1/dense_features/color_embedding/embedding_weights/m/Read/ReadVariableOpReadVariableOpDAdam/sequential_1/dense_features/color_embedding/embedding_weights/m*
_output_shapes

:*
dtype0
?
"Adam/sequential_2/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/sequential_2/dense_4/kernel/m
?
6Adam/sequential_2/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/sequential_2/dense_4/kernel/m*
_output_shapes
:	?*
dtype0
?
 Adam/sequential_2/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/sequential_2/dense_4/bias/m
?
4Adam/sequential_2/dense_4/bias/m/Read/ReadVariableOpReadVariableOp Adam/sequential_2/dense_4/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/sequential_2/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/sequential_2/dense_5/kernel/m
?
6Adam/sequential_2/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/sequential_2/dense_5/kernel/m*
_output_shapes
:	?*
dtype0
?
 Adam/sequential_2/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/sequential_2/dense_5/bias/m
?
4Adam/sequential_2/dense_5/bias/m/Read/ReadVariableOpReadVariableOp Adam/sequential_2/dense_5/bias/m*
_output_shapes
:*
dtype0
?
FAdam/sequential_1/dense_features/clarity_embedding/embedding_weights/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*W
shared_nameHFAdam/sequential_1/dense_features/clarity_embedding/embedding_weights/v
?
ZAdam/sequential_1/dense_features/clarity_embedding/embedding_weights/v/Read/ReadVariableOpReadVariableOpFAdam/sequential_1/dense_features/clarity_embedding/embedding_weights/v*
_output_shapes

:*
dtype0
?
DAdam/sequential_1/dense_features/color_embedding/embedding_weights/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*U
shared_nameFDAdam/sequential_1/dense_features/color_embedding/embedding_weights/v
?
XAdam/sequential_1/dense_features/color_embedding/embedding_weights/v/Read/ReadVariableOpReadVariableOpDAdam/sequential_1/dense_features/color_embedding/embedding_weights/v*
_output_shapes

:*
dtype0
?
"Adam/sequential_2/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/sequential_2/dense_4/kernel/v
?
6Adam/sequential_2/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/sequential_2/dense_4/kernel/v*
_output_shapes
:	?*
dtype0
?
 Adam/sequential_2/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/sequential_2/dense_4/bias/v
?
4Adam/sequential_2/dense_4/bias/v/Read/ReadVariableOpReadVariableOp Adam/sequential_2/dense_4/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/sequential_2/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/sequential_2/dense_5/kernel/v
?
6Adam/sequential_2/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/sequential_2/dense_5/kernel/v*
_output_shapes
:	?*
dtype0
?
 Adam/sequential_2/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/sequential_2/dense_5/bias/v
?
4Adam/sequential_2/dense_5/bias/v/Read/ReadVariableOpReadVariableOp Adam/sequential_2/dense_5/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
Const_2Const*
_output_shapes
:*
dtype0	*m
valuedBb	"X                                                                	       
       
?
Const_3Const*
_output_shapes
:*
dtype0	*m
valuedBb	"X                                                                	       
       
?
Const_4Const*
_output_shapes
:*
dtype0	*m
valuedBb	"X                                                                	       
       
?
Const_5Const*
_output_shapes
:*
dtype0	*m
valuedBb	"X                                                                	       
       
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_2Const_3*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_30806
?
StatefulPartitionedCall_1StatefulPartitionedCallhash_table_1Const_4Const_5*
Tin
2		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_30814
B
NoOpNoOp^StatefulPartitionedCall^StatefulPartitionedCall_1
?5
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?4
value?4B?4 B?4
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
_build_input_shape
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
?
_feature_columns

_resources
'#clarity_embedding/embedding_weights
%!color_embedding/embedding_weights
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*
?
(iter

)beta_1

*beta_2
	+decay
,learning_ratemXmYmZm[ m\!m]v^v_v`va vb!vc*
* 
.
0
1
2
3
 4
!5*
.
0
1
2
3
 4
!5*
* 
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

2serving_default* 
* 

3clarity
	4color* 
??
VARIABLE_VALUE?sequential_1/dense_features/clarity_embedding/embedding_weightsTlayer_with_weights-0/clarity_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE=sequential_1/dense_features/color_embedding/embedding_weightsRlayer_with_weights-0/color_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
ke
VARIABLE_VALUEsequential_2/dense_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEsequential_2/dense_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
?
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
ke
VARIABLE_VALUEsequential_2/dense_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEsequential_2/dense_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

 0
!1*

 0
!1*
* 
?
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

D0
E1*
* 
* 
* 

Fclarity_lookup* 

Gcolor_lookup* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	Htotal
	Icount
J	variables
K	keras_api*
8
	Ltotal
	Mcount
N	variables
O	keras_api*
R
P_initializer
Q_create_resource
R_initialize
S_destroy_resource* 
R
T_initializer
U_create_resource
V_initialize
W_destroy_resource* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

H0
I1*

J	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

L0
M1*

N	variables*
* 
* 
* 
* 
* 
* 
* 
* 
??
VARIABLE_VALUEFAdam/sequential_1/dense_features/clarity_embedding/embedding_weights/mplayer_with_weights-0/clarity_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEDAdam/sequential_1/dense_features/color_embedding/embedding_weights/mnlayer_with_weights-0/color_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/sequential_2/dense_4/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/sequential_2/dense_4/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/sequential_2/dense_5/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/sequential_2/dense_5/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEFAdam/sequential_1/dense_features/clarity_embedding/embedding_weights/vplayer_with_weights-0/clarity_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEDAdam/sequential_1/dense_features/color_embedding/embedding_weights/vnlayer_with_weights-0/color_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/sequential_2/dense_4/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/sequential_2/dense_4/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/sequential_2/dense_5/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/sequential_2/dense_5/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
p
serving_default_caratPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
r
serving_default_clarityPlaceholder*#
_output_shapes
:?????????*
dtype0	*
shape:?????????
p
serving_default_colorPlaceholder*#
_output_shapes
:?????????*
dtype0	*
shape:?????????
n
serving_default_cutPlaceholder*#
_output_shapes
:?????????*
dtype0	*
shape:?????????
p
serving_default_depthPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
p
serving_default_tablePlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
l
serving_default_xPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
l
serving_default_yPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
l
serving_default_zPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_2StatefulPartitionedCallserving_default_caratserving_default_clarityserving_default_colorserving_default_cutserving_default_depthserving_default_tableserving_default_xserving_default_yserving_default_z
hash_tableConst?sequential_1/dense_features/clarity_embedding/embedding_weightshash_table_1Const_1=sequential_1/dense_features/color_embedding/embedding_weightssequential_2/dense_4/kernelsequential_2/dense_4/biassequential_2/dense_5/kernelsequential_2/dense_5/bias*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_30301
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameSsequential_1/dense_features/clarity_embedding/embedding_weights/Read/ReadVariableOpQsequential_1/dense_features/color_embedding/embedding_weights/Read/ReadVariableOp/sequential_2/dense_4/kernel/Read/ReadVariableOp-sequential_2/dense_4/bias/Read/ReadVariableOp/sequential_2/dense_5/kernel/Read/ReadVariableOp-sequential_2/dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpZAdam/sequential_1/dense_features/clarity_embedding/embedding_weights/m/Read/ReadVariableOpXAdam/sequential_1/dense_features/color_embedding/embedding_weights/m/Read/ReadVariableOp6Adam/sequential_2/dense_4/kernel/m/Read/ReadVariableOp4Adam/sequential_2/dense_4/bias/m/Read/ReadVariableOp6Adam/sequential_2/dense_5/kernel/m/Read/ReadVariableOp4Adam/sequential_2/dense_5/bias/m/Read/ReadVariableOpZAdam/sequential_1/dense_features/clarity_embedding/embedding_weights/v/Read/ReadVariableOpXAdam/sequential_1/dense_features/color_embedding/embedding_weights/v/Read/ReadVariableOp6Adam/sequential_2/dense_4/kernel/v/Read/ReadVariableOp4Adam/sequential_2/dense_4/bias/v/Read/ReadVariableOp6Adam/sequential_2/dense_5/kernel/v/Read/ReadVariableOp4Adam/sequential_2/dense_5/bias/v/Read/ReadVariableOpConst_6*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_30936
?	
StatefulPartitionedCall_4StatefulPartitionedCallsaver_filename?sequential_1/dense_features/clarity_embedding/embedding_weights=sequential_1/dense_features/color_embedding/embedding_weightssequential_2/dense_4/kernelsequential_2/dense_4/biassequential_2/dense_5/kernelsequential_2/dense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1FAdam/sequential_1/dense_features/clarity_embedding/embedding_weights/mDAdam/sequential_1/dense_features/color_embedding/embedding_weights/m"Adam/sequential_2/dense_4/kernel/m Adam/sequential_2/dense_4/bias/m"Adam/sequential_2/dense_5/kernel/m Adam/sequential_2/dense_5/bias/mFAdam/sequential_1/dense_features/clarity_embedding/embedding_weights/vDAdam/sequential_1/dense_features/color_embedding/embedding_weights/v"Adam/sequential_2/dense_4/kernel/v Adam/sequential_2/dense_4/bias/v"Adam/sequential_2/dense_5/kernel/v Adam/sequential_2/dense_5/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_31027ˢ
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_29670

inputs
inputs_1	
inputs_2	
inputs_3	
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
dense_features_29646
dense_features_29648	&
dense_features_29650:
dense_features_29652
dense_features_29654	&
dense_features_29656: 
dense_4_29659:	?
dense_4_29661:	? 
dense_5_29664:	?
dense_5_29666:
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?&dense_features/StatefulPartitionedCall?
&dense_features/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8dense_features_29646dense_features_29648dense_features_29650dense_features_29652dense_features_29654dense_features_29656*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_29577?
dense_4/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0dense_4_29659dense_4_29661*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_29289?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_29664dense_5_29666*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_29305w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:


_output_shapes
: :

_output_shapes
: 
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_29312

inputs
inputs_1	
inputs_2	
inputs_3	
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
dense_features_29265
dense_features_29267	&
dense_features_29269:
dense_features_29271
dense_features_29273	&
dense_features_29275: 
dense_4_29290:	?
dense_4_29292:	? 
dense_5_29306:	?
dense_5_29308:
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?&dense_features/StatefulPartitionedCall?
&dense_features/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8dense_features_29265dense_features_29267dense_features_29269dense_features_29271dense_features_29273dense_features_29275*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_29264?
dense_4/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0dense_4_29290dense_4_29292*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_29289?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_29306dense_5_29308*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_29305w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:KG
#
_output_shapes
:?????????
 
_user_specified_nameinputs:


_output_shapes
: :

_output_shapes
: 
?
,
__inference__destroyer_30780
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
:
__inference__creator_30785
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name558*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?t
?
!__inference__traced_restore_31027
file_prefixb
Passignvariableop_sequential_1_dense_features_clarity_embedding_embedding_weights:b
Passignvariableop_1_sequential_1_dense_features_color_embedding_embedding_weights:A
.assignvariableop_2_sequential_2_dense_4_kernel:	?;
,assignvariableop_3_sequential_2_dense_4_bias:	?A
.assignvariableop_4_sequential_2_dense_5_kernel:	?:
,assignvariableop_5_sequential_2_dense_5_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: l
Zassignvariableop_15_adam_sequential_1_dense_features_clarity_embedding_embedding_weights_m:j
Xassignvariableop_16_adam_sequential_1_dense_features_color_embedding_embedding_weights_m:I
6assignvariableop_17_adam_sequential_2_dense_4_kernel_m:	?C
4assignvariableop_18_adam_sequential_2_dense_4_bias_m:	?I
6assignvariableop_19_adam_sequential_2_dense_5_kernel_m:	?B
4assignvariableop_20_adam_sequential_2_dense_5_bias_m:l
Zassignvariableop_21_adam_sequential_1_dense_features_clarity_embedding_embedding_weights_v:j
Xassignvariableop_22_adam_sequential_1_dense_features_color_embedding_embedding_weights_v:I
6assignvariableop_23_adam_sequential_2_dense_4_kernel_v:	?C
4assignvariableop_24_adam_sequential_2_dense_4_bias_v:	?I
6assignvariableop_25_adam_sequential_2_dense_5_kernel_v:	?B
4assignvariableop_26_adam_sequential_2_dense_5_bias_v:
identity_28??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?BTlayer_with_weights-0/clarity_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/color_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/clarity_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBnlayer_with_weights-0/color_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/clarity_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBnlayer_with_weights-0/color_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpPassignvariableop_sequential_1_dense_features_clarity_embedding_embedding_weightsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpPassignvariableop_1_sequential_1_dense_features_color_embedding_embedding_weightsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_sequential_2_dense_4_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp,assignvariableop_3_sequential_2_dense_4_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp.assignvariableop_4_sequential_2_dense_5_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp,assignvariableop_5_sequential_2_dense_5_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpZassignvariableop_15_adam_sequential_1_dense_features_clarity_embedding_embedding_weights_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpXassignvariableop_16_adam_sequential_1_dense_features_color_embedding_embedding_weights_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_sequential_2_dense_4_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adam_sequential_2_dense_4_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_sequential_2_dense_5_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_sequential_2_dense_5_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpZassignvariableop_21_adam_sequential_1_dense_features_clarity_embedding_embedding_weights_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpXassignvariableop_22_adam_sequential_1_dense_features_color_embedding_embedding_weights_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp6assignvariableop_23_adam_sequential_2_dense_4_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_sequential_2_dense_4_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_sequential_2_dense_5_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_sequential_2_dense_5_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_29761	
carat
clarity		
color	
cut		
depth	
table
x
y
z
dense_features_29737
dense_features_29739	&
dense_features_29741:
dense_features_29743
dense_features_29745	&
dense_features_29747: 
dense_4_29750:	?
dense_4_29752:	? 
dense_5_29755:	?
dense_5_29757:
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?&dense_features/StatefulPartitionedCall?
&dense_features/StatefulPartitionedCallStatefulPartitionedCallcaratclaritycolorcutdepthtablexyzdense_features_29737dense_features_29739dense_features_29741dense_features_29743dense_features_29745dense_features_29747*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_29264?
dense_4/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0dense_4_29750dense_4_29752*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_29289?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_29755dense_5_29757*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_29305w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall:J F
#
_output_shapes
:?????????

_user_specified_namecarat:LH
#
_output_shapes
:?????????
!
_user_specified_name	clarity:JF
#
_output_shapes
:?????????

_user_specified_namecolor:HD
#
_output_shapes
:?????????

_user_specified_namecut:JF
#
_output_shapes
:?????????

_user_specified_namedepth:JF
#
_output_shapes
:?????????

_user_specified_nametable:FB
#
_output_shapes
:?????????

_user_specified_namex:FB
#
_output_shapes
:?????????

_user_specified_namey:FB
#
_output_shapes
:?????????

_user_specified_namez:


_output_shapes
: :

_output_shapes
: 
??
?
I__inference_dense_features_layer_call_and_return_conditional_losses_29264
features

features_1	

features_2	

features_3	

features_4

features_5

features_6

features_7

features_8@
<clarity_embedding_none_lookup_lookuptablefindv2_table_handleA
=clarity_embedding_none_lookup_lookuptablefindv2_default_value	;
)clarity_embedding_readvariableop_resource:>
:color_embedding_none_lookup_lookuptablefindv2_table_handle?
;color_embedding_none_lookup_lookuptablefindv2_default_value	9
'color_embedding_readvariableop_resource:
identity??/clarity_embedding/None_Lookup/LookupTableFindV2? clarity_embedding/ReadVariableOp?-color_embedding/None_Lookup/LookupTableFindV2?color_embedding/ReadVariableOp_
carat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????y
carat/ExpandDims
ExpandDimsfeaturescarat/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????T
carat/ShapeShapecarat/ExpandDims:output:0*
T0*
_output_shapes
:c
carat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
carat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
carat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
carat/strided_sliceStridedSlicecarat/Shape:output:0"carat/strided_slice/stack:output:0$carat/strided_slice/stack_1:output:0$carat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
carat/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
carat/Reshape/shapePackcarat/strided_slice:output:0carat/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
carat/ReshapeReshapecarat/ExpandDims:output:0carat/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????k
 clarity_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
clarity_embedding/ExpandDims
ExpandDims
features_1)clarity_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????{
0clarity_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.clarity_embedding/to_sparse_input/ignore_valueCast9clarity_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
*clarity_embedding/to_sparse_input/NotEqualNotEqual%clarity_embedding/ExpandDims:output:02clarity_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
)clarity_embedding/to_sparse_input/indicesWhere.clarity_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
(clarity_embedding/to_sparse_input/valuesGatherNd%clarity_embedding/ExpandDims:output:01clarity_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
-clarity_embedding/to_sparse_input/dense_shapeShape%clarity_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
/clarity_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2<clarity_embedding_none_lookup_lookuptablefindv2_table_handle1clarity_embedding/to_sparse_input/values:output:0=clarity_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
 clarity_embedding/ReadVariableOpReadVariableOp)clarity_embedding_readvariableop_resource*
_output_shapes

:*
dtype0?
7clarity_embedding/clarity_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
6clarity_embedding/clarity_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
1clarity_embedding/clarity_embedding_weights/SliceSlice6clarity_embedding/to_sparse_input/dense_shape:output:0@clarity_embedding/clarity_embedding_weights/Slice/begin:output:0?clarity_embedding/clarity_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:{
1clarity_embedding/clarity_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
0clarity_embedding/clarity_embedding_weights/ProdProd:clarity_embedding/clarity_embedding_weights/Slice:output:0:clarity_embedding/clarity_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ~
<clarity_embedding/clarity_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :{
9clarity_embedding/clarity_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4clarity_embedding/clarity_embedding_weights/GatherV2GatherV26clarity_embedding/to_sparse_input/dense_shape:output:0Eclarity_embedding/clarity_embedding_weights/GatherV2/indices:output:0Bclarity_embedding/clarity_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
2clarity_embedding/clarity_embedding_weights/Cast/xPack9clarity_embedding/clarity_embedding_weights/Prod:output:0=clarity_embedding/clarity_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
9clarity_embedding/clarity_embedding_weights/SparseReshapeSparseReshape1clarity_embedding/to_sparse_input/indices:index:06clarity_embedding/to_sparse_input/dense_shape:output:0;clarity_embedding/clarity_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
Bclarity_embedding/clarity_embedding_weights/SparseReshape/IdentityIdentity8clarity_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????|
:clarity_embedding/clarity_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
8clarity_embedding/clarity_embedding_weights/GreaterEqualGreaterEqualKclarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0Cclarity_embedding/clarity_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
1clarity_embedding/clarity_embedding_weights/WhereWhere<clarity_embedding/clarity_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
9clarity_embedding/clarity_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
3clarity_embedding/clarity_embedding_weights/ReshapeReshape9clarity_embedding/clarity_embedding_weights/Where:index:0Bclarity_embedding/clarity_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:?????????}
;clarity_embedding/clarity_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6clarity_embedding/clarity_embedding_weights/GatherV2_1GatherV2Jclarity_embedding/clarity_embedding_weights/SparseReshape:output_indices:0<clarity_embedding/clarity_embedding_weights/Reshape:output:0Dclarity_embedding/clarity_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????}
;clarity_embedding/clarity_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6clarity_embedding/clarity_embedding_weights/GatherV2_2GatherV2Kclarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0<clarity_embedding/clarity_embedding_weights/Reshape:output:0Dclarity_embedding/clarity_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
4clarity_embedding/clarity_embedding_weights/IdentityIdentityHclarity_embedding/clarity_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
Eclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Sclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows?clarity_embedding/clarity_embedding_weights/GatherV2_1:output:0?clarity_embedding/clarity_embedding_weights/GatherV2_2:output:0=clarity_embedding/clarity_embedding_weights/Identity:output:0Nclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
Wclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Yclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Yclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Qclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicedclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0`clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0bclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0bclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Jclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/UniqueUniquecclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
Yclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*3
_class)
'%loc:@clarity_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
Tclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2(clarity_embedding/ReadVariableOp:value:0Nclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:y:0bclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*3
_class)
'%loc:@clarity_embedding/ReadVariableOp*'
_output_shapes
:??????????
]clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity]clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
Cclarity_embedding/clarity_embedding_weights/embedding_lookup_sparseSparseSegmentMeanfclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0Pclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:idx:0Zclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
;clarity_embedding/clarity_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
5clarity_embedding/clarity_embedding_weights/Reshape_1Reshapeiclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Dclarity_embedding/clarity_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
1clarity_embedding/clarity_embedding_weights/ShapeShapeLclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
?clarity_embedding/clarity_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Aclarity_embedding/clarity_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Aclarity_embedding/clarity_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9clarity_embedding/clarity_embedding_weights/strided_sliceStridedSlice:clarity_embedding/clarity_embedding_weights/Shape:output:0Hclarity_embedding/clarity_embedding_weights/strided_slice/stack:output:0Jclarity_embedding/clarity_embedding_weights/strided_slice/stack_1:output:0Jclarity_embedding/clarity_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3clarity_embedding/clarity_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
1clarity_embedding/clarity_embedding_weights/stackPack<clarity_embedding/clarity_embedding_weights/stack/0:output:0Bclarity_embedding/clarity_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
0clarity_embedding/clarity_embedding_weights/TileTile>clarity_embedding/clarity_embedding_weights/Reshape_1:output:0:clarity_embedding/clarity_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
6clarity_embedding/clarity_embedding_weights/zeros_like	ZerosLikeLclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
+clarity_embedding/clarity_embedding_weightsSelect9clarity_embedding/clarity_embedding_weights/Tile:output:0:clarity_embedding/clarity_embedding_weights/zeros_like:y:0Lclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
2clarity_embedding/clarity_embedding_weights/Cast_1Cast6clarity_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
9clarity_embedding/clarity_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
8clarity_embedding/clarity_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
3clarity_embedding/clarity_embedding_weights/Slice_1Slice6clarity_embedding/clarity_embedding_weights/Cast_1:y:0Bclarity_embedding/clarity_embedding_weights/Slice_1/begin:output:0Aclarity_embedding/clarity_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
3clarity_embedding/clarity_embedding_weights/Shape_1Shape4clarity_embedding/clarity_embedding_weights:output:0*
T0*
_output_shapes
:?
9clarity_embedding/clarity_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
8clarity_embedding/clarity_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
3clarity_embedding/clarity_embedding_weights/Slice_2Slice<clarity_embedding/clarity_embedding_weights/Shape_1:output:0Bclarity_embedding/clarity_embedding_weights/Slice_2/begin:output:0Aclarity_embedding/clarity_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:y
7clarity_embedding/clarity_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2clarity_embedding/clarity_embedding_weights/concatConcatV2<clarity_embedding/clarity_embedding_weights/Slice_1:output:0<clarity_embedding/clarity_embedding_weights/Slice_2:output:0@clarity_embedding/clarity_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
5clarity_embedding/clarity_embedding_weights/Reshape_2Reshape4clarity_embedding/clarity_embedding_weights:output:0;clarity_embedding/clarity_embedding_weights/concat:output:0*
T0*'
_output_shapes
:??????????
clarity_embedding/ShapeShape>clarity_embedding/clarity_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:o
%clarity_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'clarity_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'clarity_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
clarity_embedding/strided_sliceStridedSlice clarity_embedding/Shape:output:0.clarity_embedding/strided_slice/stack:output:00clarity_embedding/strided_slice/stack_1:output:00clarity_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!clarity_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
clarity_embedding/Reshape/shapePack(clarity_embedding/strided_slice:output:0*clarity_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
clarity_embedding/ReshapeReshape>clarity_embedding/clarity_embedding_weights/Reshape_2:output:0(clarity_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????i
color_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
color_embedding/ExpandDims
ExpandDims
features_2'color_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????y
.color_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
,color_embedding/to_sparse_input/ignore_valueCast7color_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
(color_embedding/to_sparse_input/NotEqualNotEqual#color_embedding/ExpandDims:output:00color_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
'color_embedding/to_sparse_input/indicesWhere,color_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
&color_embedding/to_sparse_input/valuesGatherNd#color_embedding/ExpandDims:output:0/color_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
+color_embedding/to_sparse_input/dense_shapeShape#color_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
-color_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2:color_embedding_none_lookup_lookuptablefindv2_table_handle/color_embedding/to_sparse_input/values:output:0;color_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
color_embedding/ReadVariableOpReadVariableOp'color_embedding_readvariableop_resource*
_output_shapes

:*
dtype0}
3color_embedding/color_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: |
2color_embedding/color_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
-color_embedding/color_embedding_weights/SliceSlice4color_embedding/to_sparse_input/dense_shape:output:0<color_embedding/color_embedding_weights/Slice/begin:output:0;color_embedding/color_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:w
-color_embedding/color_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
,color_embedding/color_embedding_weights/ProdProd6color_embedding/color_embedding_weights/Slice:output:06color_embedding/color_embedding_weights/Const:output:0*
T0	*
_output_shapes
: z
8color_embedding/color_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :w
5color_embedding/color_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0color_embedding/color_embedding_weights/GatherV2GatherV24color_embedding/to_sparse_input/dense_shape:output:0Acolor_embedding/color_embedding_weights/GatherV2/indices:output:0>color_embedding/color_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
.color_embedding/color_embedding_weights/Cast/xPack5color_embedding/color_embedding_weights/Prod:output:09color_embedding/color_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
5color_embedding/color_embedding_weights/SparseReshapeSparseReshape/color_embedding/to_sparse_input/indices:index:04color_embedding/to_sparse_input/dense_shape:output:07color_embedding/color_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
>color_embedding/color_embedding_weights/SparseReshape/IdentityIdentity6color_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????x
6color_embedding/color_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
4color_embedding/color_embedding_weights/GreaterEqualGreaterEqualGcolor_embedding/color_embedding_weights/SparseReshape/Identity:output:0?color_embedding/color_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
-color_embedding/color_embedding_weights/WhereWhere8color_embedding/color_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
5color_embedding/color_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
/color_embedding/color_embedding_weights/ReshapeReshape5color_embedding/color_embedding_weights/Where:index:0>color_embedding/color_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:?????????y
7color_embedding/color_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2color_embedding/color_embedding_weights/GatherV2_1GatherV2Fcolor_embedding/color_embedding_weights/SparseReshape:output_indices:08color_embedding/color_embedding_weights/Reshape:output:0@color_embedding/color_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????y
7color_embedding/color_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2color_embedding/color_embedding_weights/GatherV2_2GatherV2Gcolor_embedding/color_embedding_weights/SparseReshape/Identity:output:08color_embedding/color_embedding_weights/Reshape:output:0@color_embedding/color_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
0color_embedding/color_embedding_weights/IdentityIdentityDcolor_embedding/color_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
Acolor_embedding/color_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Ocolor_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows;color_embedding/color_embedding_weights/GatherV2_1:output:0;color_embedding/color_embedding_weights/GatherV2_2:output:09color_embedding/color_embedding_weights/Identity:output:0Jcolor_embedding/color_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
Scolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Ucolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Ucolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Mcolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice`color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0\color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0^color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0^color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Fcolor_embedding/color_embedding_weights/embedding_lookup_sparse/UniqueUnique_color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
Ucolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*1
_class'
%#loc:@color_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
Pcolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2&color_embedding/ReadVariableOp:value:0Jcolor_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:y:0^color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_class'
%#loc:@color_embedding/ReadVariableOp*'
_output_shapes
:??????????
Ycolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityYcolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
?color_embedding/color_embedding_weights/embedding_lookup_sparseSparseSegmentMeanbcolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0Lcolor_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:idx:0Vcolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
7color_embedding/color_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
1color_embedding/color_embedding_weights/Reshape_1Reshapeecolor_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0@color_embedding/color_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
-color_embedding/color_embedding_weights/ShapeShapeHcolor_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
;color_embedding/color_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
=color_embedding/color_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=color_embedding/color_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5color_embedding/color_embedding_weights/strided_sliceStridedSlice6color_embedding/color_embedding_weights/Shape:output:0Dcolor_embedding/color_embedding_weights/strided_slice/stack:output:0Fcolor_embedding/color_embedding_weights/strided_slice/stack_1:output:0Fcolor_embedding/color_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/color_embedding/color_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
-color_embedding/color_embedding_weights/stackPack8color_embedding/color_embedding_weights/stack/0:output:0>color_embedding/color_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
,color_embedding/color_embedding_weights/TileTile:color_embedding/color_embedding_weights/Reshape_1:output:06color_embedding/color_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
2color_embedding/color_embedding_weights/zeros_like	ZerosLikeHcolor_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
'color_embedding/color_embedding_weightsSelect5color_embedding/color_embedding_weights/Tile:output:06color_embedding/color_embedding_weights/zeros_like:y:0Hcolor_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
.color_embedding/color_embedding_weights/Cast_1Cast4color_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:
5color_embedding/color_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ~
4color_embedding/color_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
/color_embedding/color_embedding_weights/Slice_1Slice2color_embedding/color_embedding_weights/Cast_1:y:0>color_embedding/color_embedding_weights/Slice_1/begin:output:0=color_embedding/color_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
/color_embedding/color_embedding_weights/Shape_1Shape0color_embedding/color_embedding_weights:output:0*
T0*
_output_shapes
:
5color_embedding/color_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
4color_embedding/color_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
/color_embedding/color_embedding_weights/Slice_2Slice8color_embedding/color_embedding_weights/Shape_1:output:0>color_embedding/color_embedding_weights/Slice_2/begin:output:0=color_embedding/color_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:u
3color_embedding/color_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.color_embedding/color_embedding_weights/concatConcatV28color_embedding/color_embedding_weights/Slice_1:output:08color_embedding/color_embedding_weights/Slice_2:output:0<color_embedding/color_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1color_embedding/color_embedding_weights/Reshape_2Reshape0color_embedding/color_embedding_weights:output:07color_embedding/color_embedding_weights/concat:output:0*
T0*'
_output_shapes
:?????????
color_embedding/ShapeShape:color_embedding/color_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:m
#color_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%color_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%color_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
color_embedding/strided_sliceStridedSlicecolor_embedding/Shape:output:0,color_embedding/strided_slice/stack:output:0.color_embedding/strided_slice/stack_1:output:0.color_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
color_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
color_embedding/Reshape/shapePack&color_embedding/strided_slice:output:0(color_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
color_embedding/ReshapeReshape:color_embedding/color_embedding_weights/Reshape_2:output:0&color_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2carat/Reshape:output:0"clarity_embedding/Reshape:output:0 color_embedding/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp0^clarity_embedding/None_Lookup/LookupTableFindV2!^clarity_embedding/ReadVariableOp.^color_embedding/None_Lookup/LookupTableFindV2^color_embedding/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 2b
/clarity_embedding/None_Lookup/LookupTableFindV2/clarity_embedding/None_Lookup/LookupTableFindV22D
 clarity_embedding/ReadVariableOp clarity_embedding/ReadVariableOp2^
-color_embedding/None_Lookup/LookupTableFindV2-color_embedding/None_Lookup/LookupTableFindV22@
color_embedding/ReadVariableOpcolor_embedding/ReadVariableOp:M I
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:


_output_shapes
: :

_output_shapes
: 
?
?
G__inference_sequential_2_layer_call_and_return_conditional_losses_29796	
carat
clarity		
color	
cut		
depth	
table
x
y
z
dense_features_29772
dense_features_29774	&
dense_features_29776:
dense_features_29778
dense_features_29780	&
dense_features_29782: 
dense_4_29785:	?
dense_4_29787:	? 
dense_5_29790:	?
dense_5_29792:
identity??dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?&dense_features/StatefulPartitionedCall?
&dense_features/StatefulPartitionedCallStatefulPartitionedCallcaratclaritycolorcutdepthtablexyzdense_features_29772dense_features_29774dense_features_29776dense_features_29778dense_features_29780dense_features_29782*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_29577?
dense_4/StatefulPartitionedCallStatefulPartitionedCall/dense_features/StatefulPartitionedCall:output:0dense_4_29785dense_4_29787*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_29289?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_29790dense_5_29792*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_29305w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall'^dense_features/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2P
&dense_features/StatefulPartitionedCall&dense_features/StatefulPartitionedCall:J F
#
_output_shapes
:?????????

_user_specified_namecarat:LH
#
_output_shapes
:?????????
!
_user_specified_name	clarity:JF
#
_output_shapes
:?????????

_user_specified_namecolor:HD
#
_output_shapes
:?????????

_user_specified_namecut:JF
#
_output_shapes
:?????????

_user_specified_namedepth:JF
#
_output_shapes
:?????????

_user_specified_nametable:FB
#
_output_shapes
:?????????

_user_specified_namex:FB
#
_output_shapes
:?????????

_user_specified_namey:FB
#
_output_shapes
:?????????

_user_specified_namez:


_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_2_layer_call_fn_29868
inputs_carat
inputs_clarity	
inputs_color	

inputs_cut	
inputs_depth
inputs_table
inputs_x
inputs_y
inputs_z
unknown
	unknown_0	
	unknown_1:
	unknown_2
	unknown_3	
	unknown_4:
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_caratinputs_clarityinputs_color
inputs_cutinputs_depthinputs_tableinputs_xinputs_yinputs_zunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_29670o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
#
_output_shapes
:?????????
&
_user_specified_nameinputs/carat:SO
#
_output_shapes
:?????????
(
_user_specified_nameinputs/clarity:QM
#
_output_shapes
:?????????
&
_user_specified_nameinputs/color:OK
#
_output_shapes
:?????????
$
_user_specified_name
inputs/cut:QM
#
_output_shapes
:?????????
&
_user_specified_nameinputs/depth:QM
#
_output_shapes
:?????????
&
_user_specified_nameinputs/table:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/x:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/y:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/z:


_output_shapes
: :

_output_shapes
: 
?	
?
B__inference_dense_5_layer_call_and_return_conditional_losses_29305

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_30301	
carat
clarity		
color	
cut		
depth	
table
x
y
z
unknown
	unknown_0	
	unknown_1:
	unknown_2
	unknown_3	
	unknown_4:
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcaratclaritycolorcutdepthtablexyzunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_29055o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
#
_output_shapes
:?????????

_user_specified_namecarat:LH
#
_output_shapes
:?????????
!
_user_specified_name	clarity:JF
#
_output_shapes
:?????????

_user_specified_namecolor:HD
#
_output_shapes
:?????????

_user_specified_namecut:JF
#
_output_shapes
:?????????

_user_specified_namedepth:JF
#
_output_shapes
:?????????

_user_specified_nametable:FB
#
_output_shapes
:?????????

_user_specified_namex:FB
#
_output_shapes
:?????????

_user_specified_namey:FB
#
_output_shapes
:?????????

_user_specified_namez:


_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_2_layer_call_fn_29726	
carat
clarity		
color	
cut		
depth	
table
x
y
z
unknown
	unknown_0	
	unknown_1:
	unknown_2
	unknown_3	
	unknown_4:
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcaratclaritycolorcutdepthtablexyzunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_29670o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
#
_output_shapes
:?????????

_user_specified_namecarat:LH
#
_output_shapes
:?????????
!
_user_specified_name	clarity:JF
#
_output_shapes
:?????????

_user_specified_namecolor:HD
#
_output_shapes
:?????????

_user_specified_namecut:JF
#
_output_shapes
:?????????

_user_specified_namedepth:JF
#
_output_shapes
:?????????

_user_specified_nametable:FB
#
_output_shapes
:?????????

_user_specified_namex:FB
#
_output_shapes
:?????????

_user_specified_namey:FB
#
_output_shapes
:?????????

_user_specified_namez:


_output_shapes
: :

_output_shapes
: 
?	
?
B__inference_dense_5_layer_call_and_return_conditional_losses_30762

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_dense_features_layer_call_fn_30351
features_carat
features_clarity	
features_color	
features_cut	
features_depth
features_table

features_x

features_y

features_z
unknown
	unknown_0	
	unknown_1:
	unknown_2
	unknown_3	
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeatures_caratfeatures_clarityfeatures_colorfeatures_cutfeatures_depthfeatures_table
features_x
features_y
features_zunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_29577o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
#
_output_shapes
:?????????
(
_user_specified_namefeatures/carat:UQ
#
_output_shapes
:?????????
*
_user_specified_namefeatures/clarity:SO
#
_output_shapes
:?????????
(
_user_specified_namefeatures/color:QM
#
_output_shapes
:?????????
&
_user_specified_namefeatures/cut:SO
#
_output_shapes
:?????????
(
_user_specified_namefeatures/depth:SO
#
_output_shapes
:?????????
(
_user_specified_namefeatures/table:OK
#
_output_shapes
:?????????
$
_user_specified_name
features/x:OK
#
_output_shapes
:?????????
$
_user_specified_name
features/y:OK
#
_output_shapes
:?????????
$
_user_specified_name
features/z:


_output_shapes
: :

_output_shapes
: 
?
?
__inference__initializer_307932
.table_init557_lookuptableimportv2_table_handle*
&table_init557_lookuptableimportv2_keys	,
(table_init557_lookuptableimportv2_values	
identity??!table_init557/LookupTableImportV2?
!table_init557/LookupTableImportV2LookupTableImportV2.table_init557_lookuptableimportv2_table_handle&table_init557_lookuptableimportv2_keys(table_init557_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init557/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init557/LookupTableImportV2!table_init557/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
'__inference_dense_4_layer_call_fn_30732

inputs
unknown:	?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_29289p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?

G__inference_sequential_2_layer_call_and_return_conditional_losses_30266
inputs_carat
inputs_clarity	
inputs_color	

inputs_cut	
inputs_depth
inputs_table
inputs_x
inputs_y
inputs_zO
Kdense_features_clarity_embedding_none_lookup_lookuptablefindv2_table_handleP
Ldense_features_clarity_embedding_none_lookup_lookuptablefindv2_default_value	J
8dense_features_clarity_embedding_readvariableop_resource:M
Idense_features_color_embedding_none_lookup_lookuptablefindv2_table_handleN
Jdense_features_color_embedding_none_lookup_lookuptablefindv2_default_value	H
6dense_features_color_embedding_readvariableop_resource:9
&dense_4_matmul_readvariableop_resource:	?6
'dense_4_biasadd_readvariableop_resource:	?9
&dense_5_matmul_readvariableop_resource:	?5
'dense_5_biasadd_readvariableop_resource:
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?>dense_features/clarity_embedding/None_Lookup/LookupTableFindV2?/dense_features/clarity_embedding/ReadVariableOp?<dense_features/color_embedding/None_Lookup/LookupTableFindV2?-dense_features/color_embedding/ReadVariableOpn
#dense_features/carat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
dense_features/carat/ExpandDims
ExpandDimsinputs_carat,dense_features/carat/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????r
dense_features/carat/ShapeShape(dense_features/carat/ExpandDims:output:0*
T0*
_output_shapes
:r
(dense_features/carat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*dense_features/carat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*dense_features/carat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"dense_features/carat/strided_sliceStridedSlice#dense_features/carat/Shape:output:01dense_features/carat/strided_slice/stack:output:03dense_features/carat/strided_slice/stack_1:output:03dense_features/carat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$dense_features/carat/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"dense_features/carat/Reshape/shapePack+dense_features/carat/strided_slice:output:0-dense_features/carat/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/carat/ReshapeReshape(dense_features/carat/ExpandDims:output:0+dense_features/carat/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????z
/dense_features/clarity_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+dense_features/clarity_embedding/ExpandDims
ExpandDimsinputs_clarity8dense_features/clarity_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
?dense_features/clarity_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
=dense_features/clarity_embedding/to_sparse_input/ignore_valueCastHdense_features/clarity_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
9dense_features/clarity_embedding/to_sparse_input/NotEqualNotEqual4dense_features/clarity_embedding/ExpandDims:output:0Adense_features/clarity_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
8dense_features/clarity_embedding/to_sparse_input/indicesWhere=dense_features/clarity_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
7dense_features/clarity_embedding/to_sparse_input/valuesGatherNd4dense_features/clarity_embedding/ExpandDims:output:0@dense_features/clarity_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
<dense_features/clarity_embedding/to_sparse_input/dense_shapeShape4dense_features/clarity_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
>dense_features/clarity_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2Kdense_features_clarity_embedding_none_lookup_lookuptablefindv2_table_handle@dense_features/clarity_embedding/to_sparse_input/values:output:0Ldense_features_clarity_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
/dense_features/clarity_embedding/ReadVariableOpReadVariableOp8dense_features_clarity_embedding_readvariableop_resource*
_output_shapes

:*
dtype0?
Fdense_features/clarity_embedding/clarity_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Edense_features/clarity_embedding/clarity_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
@dense_features/clarity_embedding/clarity_embedding_weights/SliceSliceEdense_features/clarity_embedding/to_sparse_input/dense_shape:output:0Odense_features/clarity_embedding/clarity_embedding_weights/Slice/begin:output:0Ndense_features/clarity_embedding/clarity_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:?
@dense_features/clarity_embedding/clarity_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
?dense_features/clarity_embedding/clarity_embedding_weights/ProdProdIdense_features/clarity_embedding/clarity_embedding_weights/Slice:output:0Idense_features/clarity_embedding/clarity_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ?
Kdense_features/clarity_embedding/clarity_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Hdense_features/clarity_embedding/clarity_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Cdense_features/clarity_embedding/clarity_embedding_weights/GatherV2GatherV2Edense_features/clarity_embedding/to_sparse_input/dense_shape:output:0Tdense_features/clarity_embedding/clarity_embedding_weights/GatherV2/indices:output:0Qdense_features/clarity_embedding/clarity_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
Adense_features/clarity_embedding/clarity_embedding_weights/Cast/xPackHdense_features/clarity_embedding/clarity_embedding_weights/Prod:output:0Ldense_features/clarity_embedding/clarity_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
Hdense_features/clarity_embedding/clarity_embedding_weights/SparseReshapeSparseReshape@dense_features/clarity_embedding/to_sparse_input/indices:index:0Edense_features/clarity_embedding/to_sparse_input/dense_shape:output:0Jdense_features/clarity_embedding/clarity_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
Qdense_features/clarity_embedding/clarity_embedding_weights/SparseReshape/IdentityIdentityGdense_features/clarity_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
Idense_features/clarity_embedding/clarity_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Gdense_features/clarity_embedding/clarity_embedding_weights/GreaterEqualGreaterEqualZdense_features/clarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0Rdense_features/clarity_embedding/clarity_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
@dense_features/clarity_embedding/clarity_embedding_weights/WhereWhereKdense_features/clarity_embedding/clarity_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
Hdense_features/clarity_embedding/clarity_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Bdense_features/clarity_embedding/clarity_embedding_weights/ReshapeReshapeHdense_features/clarity_embedding/clarity_embedding_weights/Where:index:0Qdense_features/clarity_embedding/clarity_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:??????????
Jdense_features/clarity_embedding/clarity_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Edense_features/clarity_embedding/clarity_embedding_weights/GatherV2_1GatherV2Ydense_features/clarity_embedding/clarity_embedding_weights/SparseReshape:output_indices:0Kdense_features/clarity_embedding/clarity_embedding_weights/Reshape:output:0Sdense_features/clarity_embedding/clarity_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:??????????
Jdense_features/clarity_embedding/clarity_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Edense_features/clarity_embedding/clarity_embedding_weights/GatherV2_2GatherV2Zdense_features/clarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0Kdense_features/clarity_embedding/clarity_embedding_weights/Reshape:output:0Sdense_features/clarity_embedding/clarity_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Cdense_features/clarity_embedding/clarity_embedding_weights/IdentityIdentityWdense_features/clarity_embedding/clarity_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
Tdense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
bdense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsNdense_features/clarity_embedding/clarity_embedding_weights/GatherV2_1:output:0Ndense_features/clarity_embedding/clarity_embedding_weights/GatherV2_2:output:0Ldense_features/clarity_embedding/clarity_embedding_weights/Identity:output:0]dense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
fdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
hdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
hdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
`dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicesdense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0odense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0qdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0qdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Ydense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/UniqueUniquerdense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
hdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*B
_class8
64loc:@dense_features/clarity_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
cdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV27dense_features/clarity_embedding/ReadVariableOp:value:0]dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:y:0qdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*B
_class8
64loc:@dense_features/clarity_embedding/ReadVariableOp*'
_output_shapes
:??????????
ldense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityldense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
Rdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparseSparseSegmentMeanudense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0_dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:idx:0idense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
Jdense_features/clarity_embedding/clarity_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Ddense_features/clarity_embedding/clarity_embedding_weights/Reshape_1Reshapexdense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Sdense_features/clarity_embedding/clarity_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
@dense_features/clarity_embedding/clarity_embedding_weights/ShapeShape[dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
Ndense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Pdense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Pdense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hdense_features/clarity_embedding/clarity_embedding_weights/strided_sliceStridedSliceIdense_features/clarity_embedding/clarity_embedding_weights/Shape:output:0Wdense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stack:output:0Ydense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stack_1:output:0Ydense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Bdense_features/clarity_embedding/clarity_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
@dense_features/clarity_embedding/clarity_embedding_weights/stackPackKdense_features/clarity_embedding/clarity_embedding_weights/stack/0:output:0Qdense_features/clarity_embedding/clarity_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
?dense_features/clarity_embedding/clarity_embedding_weights/TileTileMdense_features/clarity_embedding/clarity_embedding_weights/Reshape_1:output:0Idense_features/clarity_embedding/clarity_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
Edense_features/clarity_embedding/clarity_embedding_weights/zeros_like	ZerosLike[dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
:dense_features/clarity_embedding/clarity_embedding_weightsSelectHdense_features/clarity_embedding/clarity_embedding_weights/Tile:output:0Idense_features/clarity_embedding/clarity_embedding_weights/zeros_like:y:0[dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
Adense_features/clarity_embedding/clarity_embedding_weights/Cast_1CastEdense_features/clarity_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Hdense_features/clarity_embedding/clarity_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Gdense_features/clarity_embedding/clarity_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
Bdense_features/clarity_embedding/clarity_embedding_weights/Slice_1SliceEdense_features/clarity_embedding/clarity_embedding_weights/Cast_1:y:0Qdense_features/clarity_embedding/clarity_embedding_weights/Slice_1/begin:output:0Pdense_features/clarity_embedding/clarity_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
Bdense_features/clarity_embedding/clarity_embedding_weights/Shape_1ShapeCdense_features/clarity_embedding/clarity_embedding_weights:output:0*
T0*
_output_shapes
:?
Hdense_features/clarity_embedding/clarity_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
Gdense_features/clarity_embedding/clarity_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Bdense_features/clarity_embedding/clarity_embedding_weights/Slice_2SliceKdense_features/clarity_embedding/clarity_embedding_weights/Shape_1:output:0Qdense_features/clarity_embedding/clarity_embedding_weights/Slice_2/begin:output:0Pdense_features/clarity_embedding/clarity_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:?
Fdense_features/clarity_embedding/clarity_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Adense_features/clarity_embedding/clarity_embedding_weights/concatConcatV2Kdense_features/clarity_embedding/clarity_embedding_weights/Slice_1:output:0Kdense_features/clarity_embedding/clarity_embedding_weights/Slice_2:output:0Odense_features/clarity_embedding/clarity_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Ddense_features/clarity_embedding/clarity_embedding_weights/Reshape_2ReshapeCdense_features/clarity_embedding/clarity_embedding_weights:output:0Jdense_features/clarity_embedding/clarity_embedding_weights/concat:output:0*
T0*'
_output_shapes
:??????????
&dense_features/clarity_embedding/ShapeShapeMdense_features/clarity_embedding/clarity_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:~
4dense_features/clarity_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6dense_features/clarity_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6dense_features/clarity_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.dense_features/clarity_embedding/strided_sliceStridedSlice/dense_features/clarity_embedding/Shape:output:0=dense_features/clarity_embedding/strided_slice/stack:output:0?dense_features/clarity_embedding/strided_slice/stack_1:output:0?dense_features/clarity_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0dense_features/clarity_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
.dense_features/clarity_embedding/Reshape/shapePack7dense_features/clarity_embedding/strided_slice:output:09dense_features/clarity_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
(dense_features/clarity_embedding/ReshapeReshapeMdense_features/clarity_embedding/clarity_embedding_weights/Reshape_2:output:07dense_features/clarity_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????x
-dense_features/color_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
)dense_features/color_embedding/ExpandDims
ExpandDimsinputs_color6dense_features/color_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
=dense_features/color_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
;dense_features/color_embedding/to_sparse_input/ignore_valueCastFdense_features/color_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
7dense_features/color_embedding/to_sparse_input/NotEqualNotEqual2dense_features/color_embedding/ExpandDims:output:0?dense_features/color_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
6dense_features/color_embedding/to_sparse_input/indicesWhere;dense_features/color_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
5dense_features/color_embedding/to_sparse_input/valuesGatherNd2dense_features/color_embedding/ExpandDims:output:0>dense_features/color_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
:dense_features/color_embedding/to_sparse_input/dense_shapeShape2dense_features/color_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
<dense_features/color_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2Idense_features_color_embedding_none_lookup_lookuptablefindv2_table_handle>dense_features/color_embedding/to_sparse_input/values:output:0Jdense_features_color_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
-dense_features/color_embedding/ReadVariableOpReadVariableOp6dense_features_color_embedding_readvariableop_resource*
_output_shapes

:*
dtype0?
Bdense_features/color_embedding/color_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Adense_features/color_embedding/color_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
<dense_features/color_embedding/color_embedding_weights/SliceSliceCdense_features/color_embedding/to_sparse_input/dense_shape:output:0Kdense_features/color_embedding/color_embedding_weights/Slice/begin:output:0Jdense_features/color_embedding/color_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:?
<dense_features/color_embedding/color_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
;dense_features/color_embedding/color_embedding_weights/ProdProdEdense_features/color_embedding/color_embedding_weights/Slice:output:0Edense_features/color_embedding/color_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ?
Gdense_features/color_embedding/color_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Ddense_features/color_embedding/color_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
?dense_features/color_embedding/color_embedding_weights/GatherV2GatherV2Cdense_features/color_embedding/to_sparse_input/dense_shape:output:0Pdense_features/color_embedding/color_embedding_weights/GatherV2/indices:output:0Mdense_features/color_embedding/color_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
=dense_features/color_embedding/color_embedding_weights/Cast/xPackDdense_features/color_embedding/color_embedding_weights/Prod:output:0Hdense_features/color_embedding/color_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
Ddense_features/color_embedding/color_embedding_weights/SparseReshapeSparseReshape>dense_features/color_embedding/to_sparse_input/indices:index:0Cdense_features/color_embedding/to_sparse_input/dense_shape:output:0Fdense_features/color_embedding/color_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
Mdense_features/color_embedding/color_embedding_weights/SparseReshape/IdentityIdentityEdense_features/color_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
Edense_features/color_embedding/color_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Cdense_features/color_embedding/color_embedding_weights/GreaterEqualGreaterEqualVdense_features/color_embedding/color_embedding_weights/SparseReshape/Identity:output:0Ndense_features/color_embedding/color_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
<dense_features/color_embedding/color_embedding_weights/WhereWhereGdense_features/color_embedding/color_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
Ddense_features/color_embedding/color_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
>dense_features/color_embedding/color_embedding_weights/ReshapeReshapeDdense_features/color_embedding/color_embedding_weights/Where:index:0Mdense_features/color_embedding/color_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:??????????
Fdense_features/color_embedding/color_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Adense_features/color_embedding/color_embedding_weights/GatherV2_1GatherV2Udense_features/color_embedding/color_embedding_weights/SparseReshape:output_indices:0Gdense_features/color_embedding/color_embedding_weights/Reshape:output:0Odense_features/color_embedding/color_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:??????????
Fdense_features/color_embedding/color_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Adense_features/color_embedding/color_embedding_weights/GatherV2_2GatherV2Vdense_features/color_embedding/color_embedding_weights/SparseReshape/Identity:output:0Gdense_features/color_embedding/color_embedding_weights/Reshape:output:0Odense_features/color_embedding/color_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
?dense_features/color_embedding/color_embedding_weights/IdentityIdentitySdense_features/color_embedding/color_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
Pdense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
^dense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsJdense_features/color_embedding/color_embedding_weights/GatherV2_1:output:0Jdense_features/color_embedding/color_embedding_weights/GatherV2_2:output:0Hdense_features/color_embedding/color_embedding_weights/Identity:output:0Ydense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
bdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
ddense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
ddense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
\dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceodense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0kdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0mdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0mdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Udense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/UniqueUniquendense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
ddense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*@
_class6
42loc:@dense_features/color_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
_dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV25dense_features/color_embedding/ReadVariableOp:value:0Ydense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:y:0mdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*@
_class6
42loc:@dense_features/color_embedding/ReadVariableOp*'
_output_shapes
:??????????
hdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityhdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
Ndense_features/color_embedding/color_embedding_weights/embedding_lookup_sparseSparseSegmentMeanqdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0[dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:idx:0edense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
Fdense_features/color_embedding/color_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
@dense_features/color_embedding/color_embedding_weights/Reshape_1Reshapetdense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Odense_features/color_embedding/color_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
<dense_features/color_embedding/color_embedding_weights/ShapeShapeWdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
Jdense_features/color_embedding/color_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Ldense_features/color_embedding/color_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Ldense_features/color_embedding/color_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Ddense_features/color_embedding/color_embedding_weights/strided_sliceStridedSliceEdense_features/color_embedding/color_embedding_weights/Shape:output:0Sdense_features/color_embedding/color_embedding_weights/strided_slice/stack:output:0Udense_features/color_embedding/color_embedding_weights/strided_slice/stack_1:output:0Udense_features/color_embedding/color_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
>dense_features/color_embedding/color_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
<dense_features/color_embedding/color_embedding_weights/stackPackGdense_features/color_embedding/color_embedding_weights/stack/0:output:0Mdense_features/color_embedding/color_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
;dense_features/color_embedding/color_embedding_weights/TileTileIdense_features/color_embedding/color_embedding_weights/Reshape_1:output:0Edense_features/color_embedding/color_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
Adense_features/color_embedding/color_embedding_weights/zeros_like	ZerosLikeWdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
6dense_features/color_embedding/color_embedding_weightsSelectDdense_features/color_embedding/color_embedding_weights/Tile:output:0Edense_features/color_embedding/color_embedding_weights/zeros_like:y:0Wdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
=dense_features/color_embedding/color_embedding_weights/Cast_1CastCdense_features/color_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Ddense_features/color_embedding/color_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Cdense_features/color_embedding/color_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
>dense_features/color_embedding/color_embedding_weights/Slice_1SliceAdense_features/color_embedding/color_embedding_weights/Cast_1:y:0Mdense_features/color_embedding/color_embedding_weights/Slice_1/begin:output:0Ldense_features/color_embedding/color_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
>dense_features/color_embedding/color_embedding_weights/Shape_1Shape?dense_features/color_embedding/color_embedding_weights:output:0*
T0*
_output_shapes
:?
Ddense_features/color_embedding/color_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
Cdense_features/color_embedding/color_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
>dense_features/color_embedding/color_embedding_weights/Slice_2SliceGdense_features/color_embedding/color_embedding_weights/Shape_1:output:0Mdense_features/color_embedding/color_embedding_weights/Slice_2/begin:output:0Ldense_features/color_embedding/color_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:?
Bdense_features/color_embedding/color_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
=dense_features/color_embedding/color_embedding_weights/concatConcatV2Gdense_features/color_embedding/color_embedding_weights/Slice_1:output:0Gdense_features/color_embedding/color_embedding_weights/Slice_2:output:0Kdense_features/color_embedding/color_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
@dense_features/color_embedding/color_embedding_weights/Reshape_2Reshape?dense_features/color_embedding/color_embedding_weights:output:0Fdense_features/color_embedding/color_embedding_weights/concat:output:0*
T0*'
_output_shapes
:??????????
$dense_features/color_embedding/ShapeShapeIdense_features/color_embedding/color_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:|
2dense_features/color_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4dense_features/color_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4dense_features/color_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,dense_features/color_embedding/strided_sliceStridedSlice-dense_features/color_embedding/Shape:output:0;dense_features/color_embedding/strided_slice/stack:output:0=dense_features/color_embedding/strided_slice/stack_1:output:0=dense_features/color_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.dense_features/color_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
,dense_features/color_embedding/Reshape/shapePack5dense_features/color_embedding/strided_slice:output:07dense_features/color_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
&dense_features/color_embedding/ReshapeReshapeIdense_features/color_embedding/color_embedding_weights/Reshape_2:output:05dense_features/color_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????e
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
dense_features/concatConcatV2%dense_features/carat/Reshape:output:01dense_features/clarity_embedding/Reshape:output:0/dense_features/color_embedding/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_4/MatMulMatMuldense_features/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp?^dense_features/clarity_embedding/None_Lookup/LookupTableFindV20^dense_features/clarity_embedding/ReadVariableOp=^dense_features/color_embedding/None_Lookup/LookupTableFindV2.^dense_features/color_embedding/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2?
>dense_features/clarity_embedding/None_Lookup/LookupTableFindV2>dense_features/clarity_embedding/None_Lookup/LookupTableFindV22b
/dense_features/clarity_embedding/ReadVariableOp/dense_features/clarity_embedding/ReadVariableOp2|
<dense_features/color_embedding/None_Lookup/LookupTableFindV2<dense_features/color_embedding/None_Lookup/LookupTableFindV22^
-dense_features/color_embedding/ReadVariableOp-dense_features/color_embedding/ReadVariableOp:Q M
#
_output_shapes
:?????????
&
_user_specified_nameinputs/carat:SO
#
_output_shapes
:?????????
(
_user_specified_nameinputs/clarity:QM
#
_output_shapes
:?????????
&
_user_specified_nameinputs/color:OK
#
_output_shapes
:?????????
$
_user_specified_name
inputs/cut:QM
#
_output_shapes
:?????????
&
_user_specified_nameinputs/depth:QM
#
_output_shapes
:?????????
&
_user_specified_nameinputs/table:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/x:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/y:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/z:


_output_shapes
: :

_output_shapes
: 
?
?
.__inference_dense_features_layer_call_fn_30326
features_carat
features_clarity	
features_color	
features_cut	
features_depth
features_table

features_x

features_y

features_z
unknown
	unknown_0	
	unknown_1:
	unknown_2
	unknown_3	
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeatures_caratfeatures_clarityfeatures_colorfeatures_cutfeatures_depthfeatures_table
features_x
features_y
features_zunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_29264o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
#
_output_shapes
:?????????
(
_user_specified_namefeatures/carat:UQ
#
_output_shapes
:?????????
*
_user_specified_namefeatures/clarity:SO
#
_output_shapes
:?????????
(
_user_specified_namefeatures/color:QM
#
_output_shapes
:?????????
&
_user_specified_namefeatures/cut:SO
#
_output_shapes
:?????????
(
_user_specified_namefeatures/depth:SO
#
_output_shapes
:?????????
(
_user_specified_namefeatures/table:OK
#
_output_shapes
:?????????
$
_user_specified_name
features/x:OK
#
_output_shapes
:?????????
$
_user_specified_name
features/y:OK
#
_output_shapes
:?????????
$
_user_specified_name
features/z:


_output_shapes
: :

_output_shapes
: 
?
:
__inference__creator_30767
identity??
hash_tablek

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name466*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
?
'__inference_dense_5_layer_call_fn_30752

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_29305o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
,
__inference__destroyer_30798
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference__initializer_307752
.table_init465_lookuptableimportv2_table_handle*
&table_init465_lookuptableimportv2_keys	,
(table_init465_lookuptableimportv2_values	
identity??!table_init465/LookupTableImportV2?
!table_init465/LookupTableImportV2LookupTableImportV2.table_init465_lookuptableimportv2_table_handle&table_init465_lookuptableimportv2_keys(table_init465_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init465/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init465/LookupTableImportV2!table_init465/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
__inference_<lambda>_308062
.table_init465_lookuptableimportv2_table_handle*
&table_init465_lookuptableimportv2_keys	,
(table_init465_lookuptableimportv2_values	
identity??!table_init465/LookupTableImportV2?
!table_init465/LookupTableImportV2LookupTableImportV2.table_init465_lookuptableimportv2_table_handle&table_init465_lookuptableimportv2_keys(table_init465_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init465/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init465/LookupTableImportV2!table_init465/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_30743

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?D
?
__inference__traced_save_30936
file_prefix^
Zsavev2_sequential_1_dense_features_clarity_embedding_embedding_weights_read_readvariableop\
Xsavev2_sequential_1_dense_features_color_embedding_embedding_weights_read_readvariableop:
6savev2_sequential_2_dense_4_kernel_read_readvariableop8
4savev2_sequential_2_dense_4_bias_read_readvariableop:
6savev2_sequential_2_dense_5_kernel_read_readvariableop8
4savev2_sequential_2_dense_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableope
asavev2_adam_sequential_1_dense_features_clarity_embedding_embedding_weights_m_read_readvariableopc
_savev2_adam_sequential_1_dense_features_color_embedding_embedding_weights_m_read_readvariableopA
=savev2_adam_sequential_2_dense_4_kernel_m_read_readvariableop?
;savev2_adam_sequential_2_dense_4_bias_m_read_readvariableopA
=savev2_adam_sequential_2_dense_5_kernel_m_read_readvariableop?
;savev2_adam_sequential_2_dense_5_bias_m_read_readvariableope
asavev2_adam_sequential_1_dense_features_clarity_embedding_embedding_weights_v_read_readvariableopc
_savev2_adam_sequential_1_dense_features_color_embedding_embedding_weights_v_read_readvariableopA
=savev2_adam_sequential_2_dense_4_kernel_v_read_readvariableop?
;savev2_adam_sequential_2_dense_4_bias_v_read_readvariableopA
=savev2_adam_sequential_2_dense_5_kernel_v_read_readvariableop?
;savev2_adam_sequential_2_dense_5_bias_v_read_readvariableop
savev2_const_6

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?BTlayer_with_weights-0/clarity_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/color_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/clarity_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBnlayer_with_weights-0/color_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/clarity_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBnlayer_with_weights-0/color_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Zsavev2_sequential_1_dense_features_clarity_embedding_embedding_weights_read_readvariableopXsavev2_sequential_1_dense_features_color_embedding_embedding_weights_read_readvariableop6savev2_sequential_2_dense_4_kernel_read_readvariableop4savev2_sequential_2_dense_4_bias_read_readvariableop6savev2_sequential_2_dense_5_kernel_read_readvariableop4savev2_sequential_2_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopasavev2_adam_sequential_1_dense_features_clarity_embedding_embedding_weights_m_read_readvariableop_savev2_adam_sequential_1_dense_features_color_embedding_embedding_weights_m_read_readvariableop=savev2_adam_sequential_2_dense_4_kernel_m_read_readvariableop;savev2_adam_sequential_2_dense_4_bias_m_read_readvariableop=savev2_adam_sequential_2_dense_5_kernel_m_read_readvariableop;savev2_adam_sequential_2_dense_5_bias_m_read_readvariableopasavev2_adam_sequential_1_dense_features_clarity_embedding_embedding_weights_v_read_readvariableop_savev2_adam_sequential_1_dense_features_color_embedding_embedding_weights_v_read_readvariableop=savev2_adam_sequential_2_dense_4_kernel_v_read_readvariableop;savev2_adam_sequential_2_dense_4_bias_v_read_readvariableop=savev2_adam_sequential_2_dense_5_kernel_v_read_readvariableop;savev2_adam_sequential_2_dense_5_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 **
dtypes 
2	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::	?:?:	?:: : : : : : : : : :::	?:?:	?::::	?:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

::$ 

_output_shapes

::%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

::%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

::%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_29289

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
I__inference_dense_features_layer_call_and_return_conditional_losses_30537
features_carat
features_clarity	
features_color	
features_cut	
features_depth
features_table

features_x

features_y

features_z@
<clarity_embedding_none_lookup_lookuptablefindv2_table_handleA
=clarity_embedding_none_lookup_lookuptablefindv2_default_value	;
)clarity_embedding_readvariableop_resource:>
:color_embedding_none_lookup_lookuptablefindv2_table_handle?
;color_embedding_none_lookup_lookuptablefindv2_default_value	9
'color_embedding_readvariableop_resource:
identity??/clarity_embedding/None_Lookup/LookupTableFindV2? clarity_embedding/ReadVariableOp?-color_embedding/None_Lookup/LookupTableFindV2?color_embedding/ReadVariableOp_
carat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
carat/ExpandDims
ExpandDimsfeatures_caratcarat/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????T
carat/ShapeShapecarat/ExpandDims:output:0*
T0*
_output_shapes
:c
carat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
carat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
carat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
carat/strided_sliceStridedSlicecarat/Shape:output:0"carat/strided_slice/stack:output:0$carat/strided_slice/stack_1:output:0$carat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
carat/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
carat/Reshape/shapePackcarat/strided_slice:output:0carat/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
carat/ReshapeReshapecarat/ExpandDims:output:0carat/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????k
 clarity_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
clarity_embedding/ExpandDims
ExpandDimsfeatures_clarity)clarity_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????{
0clarity_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.clarity_embedding/to_sparse_input/ignore_valueCast9clarity_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
*clarity_embedding/to_sparse_input/NotEqualNotEqual%clarity_embedding/ExpandDims:output:02clarity_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
)clarity_embedding/to_sparse_input/indicesWhere.clarity_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
(clarity_embedding/to_sparse_input/valuesGatherNd%clarity_embedding/ExpandDims:output:01clarity_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
-clarity_embedding/to_sparse_input/dense_shapeShape%clarity_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
/clarity_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2<clarity_embedding_none_lookup_lookuptablefindv2_table_handle1clarity_embedding/to_sparse_input/values:output:0=clarity_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
 clarity_embedding/ReadVariableOpReadVariableOp)clarity_embedding_readvariableop_resource*
_output_shapes

:*
dtype0?
7clarity_embedding/clarity_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
6clarity_embedding/clarity_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
1clarity_embedding/clarity_embedding_weights/SliceSlice6clarity_embedding/to_sparse_input/dense_shape:output:0@clarity_embedding/clarity_embedding_weights/Slice/begin:output:0?clarity_embedding/clarity_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:{
1clarity_embedding/clarity_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
0clarity_embedding/clarity_embedding_weights/ProdProd:clarity_embedding/clarity_embedding_weights/Slice:output:0:clarity_embedding/clarity_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ~
<clarity_embedding/clarity_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :{
9clarity_embedding/clarity_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4clarity_embedding/clarity_embedding_weights/GatherV2GatherV26clarity_embedding/to_sparse_input/dense_shape:output:0Eclarity_embedding/clarity_embedding_weights/GatherV2/indices:output:0Bclarity_embedding/clarity_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
2clarity_embedding/clarity_embedding_weights/Cast/xPack9clarity_embedding/clarity_embedding_weights/Prod:output:0=clarity_embedding/clarity_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
9clarity_embedding/clarity_embedding_weights/SparseReshapeSparseReshape1clarity_embedding/to_sparse_input/indices:index:06clarity_embedding/to_sparse_input/dense_shape:output:0;clarity_embedding/clarity_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
Bclarity_embedding/clarity_embedding_weights/SparseReshape/IdentityIdentity8clarity_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????|
:clarity_embedding/clarity_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
8clarity_embedding/clarity_embedding_weights/GreaterEqualGreaterEqualKclarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0Cclarity_embedding/clarity_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
1clarity_embedding/clarity_embedding_weights/WhereWhere<clarity_embedding/clarity_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
9clarity_embedding/clarity_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
3clarity_embedding/clarity_embedding_weights/ReshapeReshape9clarity_embedding/clarity_embedding_weights/Where:index:0Bclarity_embedding/clarity_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:?????????}
;clarity_embedding/clarity_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6clarity_embedding/clarity_embedding_weights/GatherV2_1GatherV2Jclarity_embedding/clarity_embedding_weights/SparseReshape:output_indices:0<clarity_embedding/clarity_embedding_weights/Reshape:output:0Dclarity_embedding/clarity_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????}
;clarity_embedding/clarity_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6clarity_embedding/clarity_embedding_weights/GatherV2_2GatherV2Kclarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0<clarity_embedding/clarity_embedding_weights/Reshape:output:0Dclarity_embedding/clarity_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
4clarity_embedding/clarity_embedding_weights/IdentityIdentityHclarity_embedding/clarity_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
Eclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Sclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows?clarity_embedding/clarity_embedding_weights/GatherV2_1:output:0?clarity_embedding/clarity_embedding_weights/GatherV2_2:output:0=clarity_embedding/clarity_embedding_weights/Identity:output:0Nclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
Wclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Yclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Yclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Qclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicedclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0`clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0bclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0bclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Jclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/UniqueUniquecclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
Yclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*3
_class)
'%loc:@clarity_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
Tclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2(clarity_embedding/ReadVariableOp:value:0Nclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:y:0bclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*3
_class)
'%loc:@clarity_embedding/ReadVariableOp*'
_output_shapes
:??????????
]clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity]clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
Cclarity_embedding/clarity_embedding_weights/embedding_lookup_sparseSparseSegmentMeanfclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0Pclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:idx:0Zclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
;clarity_embedding/clarity_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
5clarity_embedding/clarity_embedding_weights/Reshape_1Reshapeiclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Dclarity_embedding/clarity_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
1clarity_embedding/clarity_embedding_weights/ShapeShapeLclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
?clarity_embedding/clarity_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Aclarity_embedding/clarity_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Aclarity_embedding/clarity_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9clarity_embedding/clarity_embedding_weights/strided_sliceStridedSlice:clarity_embedding/clarity_embedding_weights/Shape:output:0Hclarity_embedding/clarity_embedding_weights/strided_slice/stack:output:0Jclarity_embedding/clarity_embedding_weights/strided_slice/stack_1:output:0Jclarity_embedding/clarity_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3clarity_embedding/clarity_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
1clarity_embedding/clarity_embedding_weights/stackPack<clarity_embedding/clarity_embedding_weights/stack/0:output:0Bclarity_embedding/clarity_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
0clarity_embedding/clarity_embedding_weights/TileTile>clarity_embedding/clarity_embedding_weights/Reshape_1:output:0:clarity_embedding/clarity_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
6clarity_embedding/clarity_embedding_weights/zeros_like	ZerosLikeLclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
+clarity_embedding/clarity_embedding_weightsSelect9clarity_embedding/clarity_embedding_weights/Tile:output:0:clarity_embedding/clarity_embedding_weights/zeros_like:y:0Lclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
2clarity_embedding/clarity_embedding_weights/Cast_1Cast6clarity_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
9clarity_embedding/clarity_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
8clarity_embedding/clarity_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
3clarity_embedding/clarity_embedding_weights/Slice_1Slice6clarity_embedding/clarity_embedding_weights/Cast_1:y:0Bclarity_embedding/clarity_embedding_weights/Slice_1/begin:output:0Aclarity_embedding/clarity_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
3clarity_embedding/clarity_embedding_weights/Shape_1Shape4clarity_embedding/clarity_embedding_weights:output:0*
T0*
_output_shapes
:?
9clarity_embedding/clarity_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
8clarity_embedding/clarity_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
3clarity_embedding/clarity_embedding_weights/Slice_2Slice<clarity_embedding/clarity_embedding_weights/Shape_1:output:0Bclarity_embedding/clarity_embedding_weights/Slice_2/begin:output:0Aclarity_embedding/clarity_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:y
7clarity_embedding/clarity_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2clarity_embedding/clarity_embedding_weights/concatConcatV2<clarity_embedding/clarity_embedding_weights/Slice_1:output:0<clarity_embedding/clarity_embedding_weights/Slice_2:output:0@clarity_embedding/clarity_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
5clarity_embedding/clarity_embedding_weights/Reshape_2Reshape4clarity_embedding/clarity_embedding_weights:output:0;clarity_embedding/clarity_embedding_weights/concat:output:0*
T0*'
_output_shapes
:??????????
clarity_embedding/ShapeShape>clarity_embedding/clarity_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:o
%clarity_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'clarity_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'clarity_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
clarity_embedding/strided_sliceStridedSlice clarity_embedding/Shape:output:0.clarity_embedding/strided_slice/stack:output:00clarity_embedding/strided_slice/stack_1:output:00clarity_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!clarity_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
clarity_embedding/Reshape/shapePack(clarity_embedding/strided_slice:output:0*clarity_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
clarity_embedding/ReshapeReshape>clarity_embedding/clarity_embedding_weights/Reshape_2:output:0(clarity_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????i
color_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
color_embedding/ExpandDims
ExpandDimsfeatures_color'color_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????y
.color_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
,color_embedding/to_sparse_input/ignore_valueCast7color_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
(color_embedding/to_sparse_input/NotEqualNotEqual#color_embedding/ExpandDims:output:00color_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
'color_embedding/to_sparse_input/indicesWhere,color_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
&color_embedding/to_sparse_input/valuesGatherNd#color_embedding/ExpandDims:output:0/color_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
+color_embedding/to_sparse_input/dense_shapeShape#color_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
-color_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2:color_embedding_none_lookup_lookuptablefindv2_table_handle/color_embedding/to_sparse_input/values:output:0;color_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
color_embedding/ReadVariableOpReadVariableOp'color_embedding_readvariableop_resource*
_output_shapes

:*
dtype0}
3color_embedding/color_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: |
2color_embedding/color_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
-color_embedding/color_embedding_weights/SliceSlice4color_embedding/to_sparse_input/dense_shape:output:0<color_embedding/color_embedding_weights/Slice/begin:output:0;color_embedding/color_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:w
-color_embedding/color_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
,color_embedding/color_embedding_weights/ProdProd6color_embedding/color_embedding_weights/Slice:output:06color_embedding/color_embedding_weights/Const:output:0*
T0	*
_output_shapes
: z
8color_embedding/color_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :w
5color_embedding/color_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0color_embedding/color_embedding_weights/GatherV2GatherV24color_embedding/to_sparse_input/dense_shape:output:0Acolor_embedding/color_embedding_weights/GatherV2/indices:output:0>color_embedding/color_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
.color_embedding/color_embedding_weights/Cast/xPack5color_embedding/color_embedding_weights/Prod:output:09color_embedding/color_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
5color_embedding/color_embedding_weights/SparseReshapeSparseReshape/color_embedding/to_sparse_input/indices:index:04color_embedding/to_sparse_input/dense_shape:output:07color_embedding/color_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
>color_embedding/color_embedding_weights/SparseReshape/IdentityIdentity6color_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????x
6color_embedding/color_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
4color_embedding/color_embedding_weights/GreaterEqualGreaterEqualGcolor_embedding/color_embedding_weights/SparseReshape/Identity:output:0?color_embedding/color_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
-color_embedding/color_embedding_weights/WhereWhere8color_embedding/color_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
5color_embedding/color_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
/color_embedding/color_embedding_weights/ReshapeReshape5color_embedding/color_embedding_weights/Where:index:0>color_embedding/color_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:?????????y
7color_embedding/color_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2color_embedding/color_embedding_weights/GatherV2_1GatherV2Fcolor_embedding/color_embedding_weights/SparseReshape:output_indices:08color_embedding/color_embedding_weights/Reshape:output:0@color_embedding/color_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????y
7color_embedding/color_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2color_embedding/color_embedding_weights/GatherV2_2GatherV2Gcolor_embedding/color_embedding_weights/SparseReshape/Identity:output:08color_embedding/color_embedding_weights/Reshape:output:0@color_embedding/color_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
0color_embedding/color_embedding_weights/IdentityIdentityDcolor_embedding/color_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
Acolor_embedding/color_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Ocolor_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows;color_embedding/color_embedding_weights/GatherV2_1:output:0;color_embedding/color_embedding_weights/GatherV2_2:output:09color_embedding/color_embedding_weights/Identity:output:0Jcolor_embedding/color_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
Scolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Ucolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Ucolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Mcolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice`color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0\color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0^color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0^color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Fcolor_embedding/color_embedding_weights/embedding_lookup_sparse/UniqueUnique_color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
Ucolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*1
_class'
%#loc:@color_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
Pcolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2&color_embedding/ReadVariableOp:value:0Jcolor_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:y:0^color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_class'
%#loc:@color_embedding/ReadVariableOp*'
_output_shapes
:??????????
Ycolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityYcolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
?color_embedding/color_embedding_weights/embedding_lookup_sparseSparseSegmentMeanbcolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0Lcolor_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:idx:0Vcolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
7color_embedding/color_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
1color_embedding/color_embedding_weights/Reshape_1Reshapeecolor_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0@color_embedding/color_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
-color_embedding/color_embedding_weights/ShapeShapeHcolor_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
;color_embedding/color_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
=color_embedding/color_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=color_embedding/color_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5color_embedding/color_embedding_weights/strided_sliceStridedSlice6color_embedding/color_embedding_weights/Shape:output:0Dcolor_embedding/color_embedding_weights/strided_slice/stack:output:0Fcolor_embedding/color_embedding_weights/strided_slice/stack_1:output:0Fcolor_embedding/color_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/color_embedding/color_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
-color_embedding/color_embedding_weights/stackPack8color_embedding/color_embedding_weights/stack/0:output:0>color_embedding/color_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
,color_embedding/color_embedding_weights/TileTile:color_embedding/color_embedding_weights/Reshape_1:output:06color_embedding/color_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
2color_embedding/color_embedding_weights/zeros_like	ZerosLikeHcolor_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
'color_embedding/color_embedding_weightsSelect5color_embedding/color_embedding_weights/Tile:output:06color_embedding/color_embedding_weights/zeros_like:y:0Hcolor_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
.color_embedding/color_embedding_weights/Cast_1Cast4color_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:
5color_embedding/color_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ~
4color_embedding/color_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
/color_embedding/color_embedding_weights/Slice_1Slice2color_embedding/color_embedding_weights/Cast_1:y:0>color_embedding/color_embedding_weights/Slice_1/begin:output:0=color_embedding/color_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
/color_embedding/color_embedding_weights/Shape_1Shape0color_embedding/color_embedding_weights:output:0*
T0*
_output_shapes
:
5color_embedding/color_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
4color_embedding/color_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
/color_embedding/color_embedding_weights/Slice_2Slice8color_embedding/color_embedding_weights/Shape_1:output:0>color_embedding/color_embedding_weights/Slice_2/begin:output:0=color_embedding/color_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:u
3color_embedding/color_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.color_embedding/color_embedding_weights/concatConcatV28color_embedding/color_embedding_weights/Slice_1:output:08color_embedding/color_embedding_weights/Slice_2:output:0<color_embedding/color_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1color_embedding/color_embedding_weights/Reshape_2Reshape0color_embedding/color_embedding_weights:output:07color_embedding/color_embedding_weights/concat:output:0*
T0*'
_output_shapes
:?????????
color_embedding/ShapeShape:color_embedding/color_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:m
#color_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%color_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%color_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
color_embedding/strided_sliceStridedSlicecolor_embedding/Shape:output:0,color_embedding/strided_slice/stack:output:0.color_embedding/strided_slice/stack_1:output:0.color_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
color_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
color_embedding/Reshape/shapePack&color_embedding/strided_slice:output:0(color_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
color_embedding/ReshapeReshape:color_embedding/color_embedding_weights/Reshape_2:output:0&color_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2carat/Reshape:output:0"clarity_embedding/Reshape:output:0 color_embedding/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp0^clarity_embedding/None_Lookup/LookupTableFindV2!^clarity_embedding/ReadVariableOp.^color_embedding/None_Lookup/LookupTableFindV2^color_embedding/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 2b
/clarity_embedding/None_Lookup/LookupTableFindV2/clarity_embedding/None_Lookup/LookupTableFindV22D
 clarity_embedding/ReadVariableOp clarity_embedding/ReadVariableOp2^
-color_embedding/None_Lookup/LookupTableFindV2-color_embedding/None_Lookup/LookupTableFindV22@
color_embedding/ReadVariableOpcolor_embedding/ReadVariableOp:S O
#
_output_shapes
:?????????
(
_user_specified_namefeatures/carat:UQ
#
_output_shapes
:?????????
*
_user_specified_namefeatures/clarity:SO
#
_output_shapes
:?????????
(
_user_specified_namefeatures/color:QM
#
_output_shapes
:?????????
&
_user_specified_namefeatures/cut:SO
#
_output_shapes
:?????????
(
_user_specified_namefeatures/depth:SO
#
_output_shapes
:?????????
(
_user_specified_namefeatures/table:OK
#
_output_shapes
:?????????
$
_user_specified_name
features/x:OK
#
_output_shapes
:?????????
$
_user_specified_name
features/y:OK
#
_output_shapes
:?????????
$
_user_specified_name
features/z:


_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_2_layer_call_fn_29835
inputs_carat
inputs_clarity	
inputs_color	

inputs_cut	
inputs_depth
inputs_table
inputs_x
inputs_y
inputs_z
unknown
	unknown_0	
	unknown_1:
	unknown_2
	unknown_3	
	unknown_4:
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_caratinputs_clarityinputs_color
inputs_cutinputs_depthinputs_tableinputs_xinputs_yinputs_zunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_29312o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
#
_output_shapes
:?????????
&
_user_specified_nameinputs/carat:SO
#
_output_shapes
:?????????
(
_user_specified_nameinputs/clarity:QM
#
_output_shapes
:?????????
&
_user_specified_nameinputs/color:OK
#
_output_shapes
:?????????
$
_user_specified_name
inputs/cut:QM
#
_output_shapes
:?????????
&
_user_specified_nameinputs/depth:QM
#
_output_shapes
:?????????
&
_user_specified_nameinputs/table:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/x:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/y:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/z:


_output_shapes
: :

_output_shapes
: 
??
?
I__inference_dense_features_layer_call_and_return_conditional_losses_29577
features

features_1	

features_2	

features_3	

features_4

features_5

features_6

features_7

features_8@
<clarity_embedding_none_lookup_lookuptablefindv2_table_handleA
=clarity_embedding_none_lookup_lookuptablefindv2_default_value	;
)clarity_embedding_readvariableop_resource:>
:color_embedding_none_lookup_lookuptablefindv2_table_handle?
;color_embedding_none_lookup_lookuptablefindv2_default_value	9
'color_embedding_readvariableop_resource:
identity??/clarity_embedding/None_Lookup/LookupTableFindV2? clarity_embedding/ReadVariableOp?-color_embedding/None_Lookup/LookupTableFindV2?color_embedding/ReadVariableOp_
carat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????y
carat/ExpandDims
ExpandDimsfeaturescarat/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????T
carat/ShapeShapecarat/ExpandDims:output:0*
T0*
_output_shapes
:c
carat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
carat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
carat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
carat/strided_sliceStridedSlicecarat/Shape:output:0"carat/strided_slice/stack:output:0$carat/strided_slice/stack_1:output:0$carat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
carat/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
carat/Reshape/shapePackcarat/strided_slice:output:0carat/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
carat/ReshapeReshapecarat/ExpandDims:output:0carat/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????k
 clarity_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
clarity_embedding/ExpandDims
ExpandDims
features_1)clarity_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????{
0clarity_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.clarity_embedding/to_sparse_input/ignore_valueCast9clarity_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
*clarity_embedding/to_sparse_input/NotEqualNotEqual%clarity_embedding/ExpandDims:output:02clarity_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
)clarity_embedding/to_sparse_input/indicesWhere.clarity_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
(clarity_embedding/to_sparse_input/valuesGatherNd%clarity_embedding/ExpandDims:output:01clarity_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
-clarity_embedding/to_sparse_input/dense_shapeShape%clarity_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
/clarity_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2<clarity_embedding_none_lookup_lookuptablefindv2_table_handle1clarity_embedding/to_sparse_input/values:output:0=clarity_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
 clarity_embedding/ReadVariableOpReadVariableOp)clarity_embedding_readvariableop_resource*
_output_shapes

:*
dtype0?
7clarity_embedding/clarity_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
6clarity_embedding/clarity_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
1clarity_embedding/clarity_embedding_weights/SliceSlice6clarity_embedding/to_sparse_input/dense_shape:output:0@clarity_embedding/clarity_embedding_weights/Slice/begin:output:0?clarity_embedding/clarity_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:{
1clarity_embedding/clarity_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
0clarity_embedding/clarity_embedding_weights/ProdProd:clarity_embedding/clarity_embedding_weights/Slice:output:0:clarity_embedding/clarity_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ~
<clarity_embedding/clarity_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :{
9clarity_embedding/clarity_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4clarity_embedding/clarity_embedding_weights/GatherV2GatherV26clarity_embedding/to_sparse_input/dense_shape:output:0Eclarity_embedding/clarity_embedding_weights/GatherV2/indices:output:0Bclarity_embedding/clarity_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
2clarity_embedding/clarity_embedding_weights/Cast/xPack9clarity_embedding/clarity_embedding_weights/Prod:output:0=clarity_embedding/clarity_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
9clarity_embedding/clarity_embedding_weights/SparseReshapeSparseReshape1clarity_embedding/to_sparse_input/indices:index:06clarity_embedding/to_sparse_input/dense_shape:output:0;clarity_embedding/clarity_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
Bclarity_embedding/clarity_embedding_weights/SparseReshape/IdentityIdentity8clarity_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????|
:clarity_embedding/clarity_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
8clarity_embedding/clarity_embedding_weights/GreaterEqualGreaterEqualKclarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0Cclarity_embedding/clarity_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
1clarity_embedding/clarity_embedding_weights/WhereWhere<clarity_embedding/clarity_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
9clarity_embedding/clarity_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
3clarity_embedding/clarity_embedding_weights/ReshapeReshape9clarity_embedding/clarity_embedding_weights/Where:index:0Bclarity_embedding/clarity_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:?????????}
;clarity_embedding/clarity_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6clarity_embedding/clarity_embedding_weights/GatherV2_1GatherV2Jclarity_embedding/clarity_embedding_weights/SparseReshape:output_indices:0<clarity_embedding/clarity_embedding_weights/Reshape:output:0Dclarity_embedding/clarity_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????}
;clarity_embedding/clarity_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6clarity_embedding/clarity_embedding_weights/GatherV2_2GatherV2Kclarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0<clarity_embedding/clarity_embedding_weights/Reshape:output:0Dclarity_embedding/clarity_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
4clarity_embedding/clarity_embedding_weights/IdentityIdentityHclarity_embedding/clarity_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
Eclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Sclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows?clarity_embedding/clarity_embedding_weights/GatherV2_1:output:0?clarity_embedding/clarity_embedding_weights/GatherV2_2:output:0=clarity_embedding/clarity_embedding_weights/Identity:output:0Nclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
Wclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Yclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Yclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Qclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicedclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0`clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0bclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0bclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Jclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/UniqueUniquecclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
Yclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*3
_class)
'%loc:@clarity_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
Tclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2(clarity_embedding/ReadVariableOp:value:0Nclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:y:0bclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*3
_class)
'%loc:@clarity_embedding/ReadVariableOp*'
_output_shapes
:??????????
]clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity]clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
Cclarity_embedding/clarity_embedding_weights/embedding_lookup_sparseSparseSegmentMeanfclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0Pclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:idx:0Zclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
;clarity_embedding/clarity_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
5clarity_embedding/clarity_embedding_weights/Reshape_1Reshapeiclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Dclarity_embedding/clarity_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
1clarity_embedding/clarity_embedding_weights/ShapeShapeLclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
?clarity_embedding/clarity_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Aclarity_embedding/clarity_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Aclarity_embedding/clarity_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9clarity_embedding/clarity_embedding_weights/strided_sliceStridedSlice:clarity_embedding/clarity_embedding_weights/Shape:output:0Hclarity_embedding/clarity_embedding_weights/strided_slice/stack:output:0Jclarity_embedding/clarity_embedding_weights/strided_slice/stack_1:output:0Jclarity_embedding/clarity_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3clarity_embedding/clarity_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
1clarity_embedding/clarity_embedding_weights/stackPack<clarity_embedding/clarity_embedding_weights/stack/0:output:0Bclarity_embedding/clarity_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
0clarity_embedding/clarity_embedding_weights/TileTile>clarity_embedding/clarity_embedding_weights/Reshape_1:output:0:clarity_embedding/clarity_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
6clarity_embedding/clarity_embedding_weights/zeros_like	ZerosLikeLclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
+clarity_embedding/clarity_embedding_weightsSelect9clarity_embedding/clarity_embedding_weights/Tile:output:0:clarity_embedding/clarity_embedding_weights/zeros_like:y:0Lclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
2clarity_embedding/clarity_embedding_weights/Cast_1Cast6clarity_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
9clarity_embedding/clarity_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
8clarity_embedding/clarity_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
3clarity_embedding/clarity_embedding_weights/Slice_1Slice6clarity_embedding/clarity_embedding_weights/Cast_1:y:0Bclarity_embedding/clarity_embedding_weights/Slice_1/begin:output:0Aclarity_embedding/clarity_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
3clarity_embedding/clarity_embedding_weights/Shape_1Shape4clarity_embedding/clarity_embedding_weights:output:0*
T0*
_output_shapes
:?
9clarity_embedding/clarity_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
8clarity_embedding/clarity_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
3clarity_embedding/clarity_embedding_weights/Slice_2Slice<clarity_embedding/clarity_embedding_weights/Shape_1:output:0Bclarity_embedding/clarity_embedding_weights/Slice_2/begin:output:0Aclarity_embedding/clarity_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:y
7clarity_embedding/clarity_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2clarity_embedding/clarity_embedding_weights/concatConcatV2<clarity_embedding/clarity_embedding_weights/Slice_1:output:0<clarity_embedding/clarity_embedding_weights/Slice_2:output:0@clarity_embedding/clarity_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
5clarity_embedding/clarity_embedding_weights/Reshape_2Reshape4clarity_embedding/clarity_embedding_weights:output:0;clarity_embedding/clarity_embedding_weights/concat:output:0*
T0*'
_output_shapes
:??????????
clarity_embedding/ShapeShape>clarity_embedding/clarity_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:o
%clarity_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'clarity_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'clarity_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
clarity_embedding/strided_sliceStridedSlice clarity_embedding/Shape:output:0.clarity_embedding/strided_slice/stack:output:00clarity_embedding/strided_slice/stack_1:output:00clarity_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!clarity_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
clarity_embedding/Reshape/shapePack(clarity_embedding/strided_slice:output:0*clarity_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
clarity_embedding/ReshapeReshape>clarity_embedding/clarity_embedding_weights/Reshape_2:output:0(clarity_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????i
color_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
color_embedding/ExpandDims
ExpandDims
features_2'color_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????y
.color_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
,color_embedding/to_sparse_input/ignore_valueCast7color_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
(color_embedding/to_sparse_input/NotEqualNotEqual#color_embedding/ExpandDims:output:00color_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
'color_embedding/to_sparse_input/indicesWhere,color_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
&color_embedding/to_sparse_input/valuesGatherNd#color_embedding/ExpandDims:output:0/color_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
+color_embedding/to_sparse_input/dense_shapeShape#color_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
-color_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2:color_embedding_none_lookup_lookuptablefindv2_table_handle/color_embedding/to_sparse_input/values:output:0;color_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
color_embedding/ReadVariableOpReadVariableOp'color_embedding_readvariableop_resource*
_output_shapes

:*
dtype0}
3color_embedding/color_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: |
2color_embedding/color_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
-color_embedding/color_embedding_weights/SliceSlice4color_embedding/to_sparse_input/dense_shape:output:0<color_embedding/color_embedding_weights/Slice/begin:output:0;color_embedding/color_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:w
-color_embedding/color_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
,color_embedding/color_embedding_weights/ProdProd6color_embedding/color_embedding_weights/Slice:output:06color_embedding/color_embedding_weights/Const:output:0*
T0	*
_output_shapes
: z
8color_embedding/color_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :w
5color_embedding/color_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0color_embedding/color_embedding_weights/GatherV2GatherV24color_embedding/to_sparse_input/dense_shape:output:0Acolor_embedding/color_embedding_weights/GatherV2/indices:output:0>color_embedding/color_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
.color_embedding/color_embedding_weights/Cast/xPack5color_embedding/color_embedding_weights/Prod:output:09color_embedding/color_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
5color_embedding/color_embedding_weights/SparseReshapeSparseReshape/color_embedding/to_sparse_input/indices:index:04color_embedding/to_sparse_input/dense_shape:output:07color_embedding/color_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
>color_embedding/color_embedding_weights/SparseReshape/IdentityIdentity6color_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????x
6color_embedding/color_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
4color_embedding/color_embedding_weights/GreaterEqualGreaterEqualGcolor_embedding/color_embedding_weights/SparseReshape/Identity:output:0?color_embedding/color_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
-color_embedding/color_embedding_weights/WhereWhere8color_embedding/color_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
5color_embedding/color_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
/color_embedding/color_embedding_weights/ReshapeReshape5color_embedding/color_embedding_weights/Where:index:0>color_embedding/color_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:?????????y
7color_embedding/color_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2color_embedding/color_embedding_weights/GatherV2_1GatherV2Fcolor_embedding/color_embedding_weights/SparseReshape:output_indices:08color_embedding/color_embedding_weights/Reshape:output:0@color_embedding/color_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????y
7color_embedding/color_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2color_embedding/color_embedding_weights/GatherV2_2GatherV2Gcolor_embedding/color_embedding_weights/SparseReshape/Identity:output:08color_embedding/color_embedding_weights/Reshape:output:0@color_embedding/color_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
0color_embedding/color_embedding_weights/IdentityIdentityDcolor_embedding/color_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
Acolor_embedding/color_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Ocolor_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows;color_embedding/color_embedding_weights/GatherV2_1:output:0;color_embedding/color_embedding_weights/GatherV2_2:output:09color_embedding/color_embedding_weights/Identity:output:0Jcolor_embedding/color_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
Scolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Ucolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Ucolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Mcolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice`color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0\color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0^color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0^color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Fcolor_embedding/color_embedding_weights/embedding_lookup_sparse/UniqueUnique_color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
Ucolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*1
_class'
%#loc:@color_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
Pcolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2&color_embedding/ReadVariableOp:value:0Jcolor_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:y:0^color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_class'
%#loc:@color_embedding/ReadVariableOp*'
_output_shapes
:??????????
Ycolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityYcolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
?color_embedding/color_embedding_weights/embedding_lookup_sparseSparseSegmentMeanbcolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0Lcolor_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:idx:0Vcolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
7color_embedding/color_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
1color_embedding/color_embedding_weights/Reshape_1Reshapeecolor_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0@color_embedding/color_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
-color_embedding/color_embedding_weights/ShapeShapeHcolor_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
;color_embedding/color_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
=color_embedding/color_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=color_embedding/color_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5color_embedding/color_embedding_weights/strided_sliceStridedSlice6color_embedding/color_embedding_weights/Shape:output:0Dcolor_embedding/color_embedding_weights/strided_slice/stack:output:0Fcolor_embedding/color_embedding_weights/strided_slice/stack_1:output:0Fcolor_embedding/color_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/color_embedding/color_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
-color_embedding/color_embedding_weights/stackPack8color_embedding/color_embedding_weights/stack/0:output:0>color_embedding/color_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
,color_embedding/color_embedding_weights/TileTile:color_embedding/color_embedding_weights/Reshape_1:output:06color_embedding/color_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
2color_embedding/color_embedding_weights/zeros_like	ZerosLikeHcolor_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
'color_embedding/color_embedding_weightsSelect5color_embedding/color_embedding_weights/Tile:output:06color_embedding/color_embedding_weights/zeros_like:y:0Hcolor_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
.color_embedding/color_embedding_weights/Cast_1Cast4color_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:
5color_embedding/color_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ~
4color_embedding/color_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
/color_embedding/color_embedding_weights/Slice_1Slice2color_embedding/color_embedding_weights/Cast_1:y:0>color_embedding/color_embedding_weights/Slice_1/begin:output:0=color_embedding/color_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
/color_embedding/color_embedding_weights/Shape_1Shape0color_embedding/color_embedding_weights:output:0*
T0*
_output_shapes
:
5color_embedding/color_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
4color_embedding/color_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
/color_embedding/color_embedding_weights/Slice_2Slice8color_embedding/color_embedding_weights/Shape_1:output:0>color_embedding/color_embedding_weights/Slice_2/begin:output:0=color_embedding/color_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:u
3color_embedding/color_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.color_embedding/color_embedding_weights/concatConcatV28color_embedding/color_embedding_weights/Slice_1:output:08color_embedding/color_embedding_weights/Slice_2:output:0<color_embedding/color_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1color_embedding/color_embedding_weights/Reshape_2Reshape0color_embedding/color_embedding_weights:output:07color_embedding/color_embedding_weights/concat:output:0*
T0*'
_output_shapes
:?????????
color_embedding/ShapeShape:color_embedding/color_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:m
#color_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%color_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%color_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
color_embedding/strided_sliceStridedSlicecolor_embedding/Shape:output:0,color_embedding/strided_slice/stack:output:0.color_embedding/strided_slice/stack_1:output:0.color_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
color_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
color_embedding/Reshape/shapePack&color_embedding/strided_slice:output:0(color_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
color_embedding/ReshapeReshape:color_embedding/color_embedding_weights/Reshape_2:output:0&color_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2carat/Reshape:output:0"clarity_embedding/Reshape:output:0 color_embedding/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp0^clarity_embedding/None_Lookup/LookupTableFindV2!^clarity_embedding/ReadVariableOp.^color_embedding/None_Lookup/LookupTableFindV2^color_embedding/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 2b
/clarity_embedding/None_Lookup/LookupTableFindV2/clarity_embedding/None_Lookup/LookupTableFindV22D
 clarity_embedding/ReadVariableOp clarity_embedding/ReadVariableOp2^
-color_embedding/None_Lookup/LookupTableFindV2-color_embedding/None_Lookup/LookupTableFindV22@
color_embedding/ReadVariableOpcolor_embedding/ReadVariableOp:M I
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:MI
#
_output_shapes
:?????????
"
_user_specified_name
features:


_output_shapes
: :

_output_shapes
: 
??
?
I__inference_dense_features_layer_call_and_return_conditional_losses_30723
features_carat
features_clarity	
features_color	
features_cut	
features_depth
features_table

features_x

features_y

features_z@
<clarity_embedding_none_lookup_lookuptablefindv2_table_handleA
=clarity_embedding_none_lookup_lookuptablefindv2_default_value	;
)clarity_embedding_readvariableop_resource:>
:color_embedding_none_lookup_lookuptablefindv2_table_handle?
;color_embedding_none_lookup_lookuptablefindv2_default_value	9
'color_embedding_readvariableop_resource:
identity??/clarity_embedding/None_Lookup/LookupTableFindV2? clarity_embedding/ReadVariableOp?-color_embedding/None_Lookup/LookupTableFindV2?color_embedding/ReadVariableOp_
carat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????
carat/ExpandDims
ExpandDimsfeatures_caratcarat/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????T
carat/ShapeShapecarat/ExpandDims:output:0*
T0*
_output_shapes
:c
carat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
carat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
carat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
carat/strided_sliceStridedSlicecarat/Shape:output:0"carat/strided_slice/stack:output:0$carat/strided_slice/stack_1:output:0$carat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
carat/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
carat/Reshape/shapePackcarat/strided_slice:output:0carat/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
carat/ReshapeReshapecarat/ExpandDims:output:0carat/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????k
 clarity_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
clarity_embedding/ExpandDims
ExpandDimsfeatures_clarity)clarity_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????{
0clarity_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
.clarity_embedding/to_sparse_input/ignore_valueCast9clarity_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
*clarity_embedding/to_sparse_input/NotEqualNotEqual%clarity_embedding/ExpandDims:output:02clarity_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
)clarity_embedding/to_sparse_input/indicesWhere.clarity_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
(clarity_embedding/to_sparse_input/valuesGatherNd%clarity_embedding/ExpandDims:output:01clarity_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
-clarity_embedding/to_sparse_input/dense_shapeShape%clarity_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
/clarity_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2<clarity_embedding_none_lookup_lookuptablefindv2_table_handle1clarity_embedding/to_sparse_input/values:output:0=clarity_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
 clarity_embedding/ReadVariableOpReadVariableOp)clarity_embedding_readvariableop_resource*
_output_shapes

:*
dtype0?
7clarity_embedding/clarity_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
6clarity_embedding/clarity_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
1clarity_embedding/clarity_embedding_weights/SliceSlice6clarity_embedding/to_sparse_input/dense_shape:output:0@clarity_embedding/clarity_embedding_weights/Slice/begin:output:0?clarity_embedding/clarity_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:{
1clarity_embedding/clarity_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
0clarity_embedding/clarity_embedding_weights/ProdProd:clarity_embedding/clarity_embedding_weights/Slice:output:0:clarity_embedding/clarity_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ~
<clarity_embedding/clarity_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :{
9clarity_embedding/clarity_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
4clarity_embedding/clarity_embedding_weights/GatherV2GatherV26clarity_embedding/to_sparse_input/dense_shape:output:0Eclarity_embedding/clarity_embedding_weights/GatherV2/indices:output:0Bclarity_embedding/clarity_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
2clarity_embedding/clarity_embedding_weights/Cast/xPack9clarity_embedding/clarity_embedding_weights/Prod:output:0=clarity_embedding/clarity_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
9clarity_embedding/clarity_embedding_weights/SparseReshapeSparseReshape1clarity_embedding/to_sparse_input/indices:index:06clarity_embedding/to_sparse_input/dense_shape:output:0;clarity_embedding/clarity_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
Bclarity_embedding/clarity_embedding_weights/SparseReshape/IdentityIdentity8clarity_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????|
:clarity_embedding/clarity_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
8clarity_embedding/clarity_embedding_weights/GreaterEqualGreaterEqualKclarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0Cclarity_embedding/clarity_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
1clarity_embedding/clarity_embedding_weights/WhereWhere<clarity_embedding/clarity_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
9clarity_embedding/clarity_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
3clarity_embedding/clarity_embedding_weights/ReshapeReshape9clarity_embedding/clarity_embedding_weights/Where:index:0Bclarity_embedding/clarity_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:?????????}
;clarity_embedding/clarity_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6clarity_embedding/clarity_embedding_weights/GatherV2_1GatherV2Jclarity_embedding/clarity_embedding_weights/SparseReshape:output_indices:0<clarity_embedding/clarity_embedding_weights/Reshape:output:0Dclarity_embedding/clarity_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????}
;clarity_embedding/clarity_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
6clarity_embedding/clarity_embedding_weights/GatherV2_2GatherV2Kclarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0<clarity_embedding/clarity_embedding_weights/Reshape:output:0Dclarity_embedding/clarity_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
4clarity_embedding/clarity_embedding_weights/IdentityIdentityHclarity_embedding/clarity_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
Eclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Sclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows?clarity_embedding/clarity_embedding_weights/GatherV2_1:output:0?clarity_embedding/clarity_embedding_weights/GatherV2_2:output:0=clarity_embedding/clarity_embedding_weights/Identity:output:0Nclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
Wclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Yclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Yclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Qclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicedclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0`clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0bclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0bclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Jclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/UniqueUniquecclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
Yclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*3
_class)
'%loc:@clarity_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
Tclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2(clarity_embedding/ReadVariableOp:value:0Nclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:y:0bclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*3
_class)
'%loc:@clarity_embedding/ReadVariableOp*'
_output_shapes
:??????????
]clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity]clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
Cclarity_embedding/clarity_embedding_weights/embedding_lookup_sparseSparseSegmentMeanfclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0Pclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:idx:0Zclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
;clarity_embedding/clarity_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
5clarity_embedding/clarity_embedding_weights/Reshape_1Reshapeiclarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Dclarity_embedding/clarity_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
1clarity_embedding/clarity_embedding_weights/ShapeShapeLclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
?clarity_embedding/clarity_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Aclarity_embedding/clarity_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Aclarity_embedding/clarity_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9clarity_embedding/clarity_embedding_weights/strided_sliceStridedSlice:clarity_embedding/clarity_embedding_weights/Shape:output:0Hclarity_embedding/clarity_embedding_weights/strided_slice/stack:output:0Jclarity_embedding/clarity_embedding_weights/strided_slice/stack_1:output:0Jclarity_embedding/clarity_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3clarity_embedding/clarity_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
1clarity_embedding/clarity_embedding_weights/stackPack<clarity_embedding/clarity_embedding_weights/stack/0:output:0Bclarity_embedding/clarity_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
0clarity_embedding/clarity_embedding_weights/TileTile>clarity_embedding/clarity_embedding_weights/Reshape_1:output:0:clarity_embedding/clarity_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
6clarity_embedding/clarity_embedding_weights/zeros_like	ZerosLikeLclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
+clarity_embedding/clarity_embedding_weightsSelect9clarity_embedding/clarity_embedding_weights/Tile:output:0:clarity_embedding/clarity_embedding_weights/zeros_like:y:0Lclarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
2clarity_embedding/clarity_embedding_weights/Cast_1Cast6clarity_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
9clarity_embedding/clarity_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
8clarity_embedding/clarity_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
3clarity_embedding/clarity_embedding_weights/Slice_1Slice6clarity_embedding/clarity_embedding_weights/Cast_1:y:0Bclarity_embedding/clarity_embedding_weights/Slice_1/begin:output:0Aclarity_embedding/clarity_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
3clarity_embedding/clarity_embedding_weights/Shape_1Shape4clarity_embedding/clarity_embedding_weights:output:0*
T0*
_output_shapes
:?
9clarity_embedding/clarity_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
8clarity_embedding/clarity_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
3clarity_embedding/clarity_embedding_weights/Slice_2Slice<clarity_embedding/clarity_embedding_weights/Shape_1:output:0Bclarity_embedding/clarity_embedding_weights/Slice_2/begin:output:0Aclarity_embedding/clarity_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:y
7clarity_embedding/clarity_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2clarity_embedding/clarity_embedding_weights/concatConcatV2<clarity_embedding/clarity_embedding_weights/Slice_1:output:0<clarity_embedding/clarity_embedding_weights/Slice_2:output:0@clarity_embedding/clarity_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
5clarity_embedding/clarity_embedding_weights/Reshape_2Reshape4clarity_embedding/clarity_embedding_weights:output:0;clarity_embedding/clarity_embedding_weights/concat:output:0*
T0*'
_output_shapes
:??????????
clarity_embedding/ShapeShape>clarity_embedding/clarity_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:o
%clarity_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'clarity_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'clarity_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
clarity_embedding/strided_sliceStridedSlice clarity_embedding/Shape:output:0.clarity_embedding/strided_slice/stack:output:00clarity_embedding/strided_slice/stack_1:output:00clarity_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!clarity_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
clarity_embedding/Reshape/shapePack(clarity_embedding/strided_slice:output:0*clarity_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
clarity_embedding/ReshapeReshape>clarity_embedding/clarity_embedding_weights/Reshape_2:output:0(clarity_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????i
color_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
color_embedding/ExpandDims
ExpandDimsfeatures_color'color_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:?????????y
.color_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
,color_embedding/to_sparse_input/ignore_valueCast7color_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
(color_embedding/to_sparse_input/NotEqualNotEqual#color_embedding/ExpandDims:output:00color_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
'color_embedding/to_sparse_input/indicesWhere,color_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
&color_embedding/to_sparse_input/valuesGatherNd#color_embedding/ExpandDims:output:0/color_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
+color_embedding/to_sparse_input/dense_shapeShape#color_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
-color_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2:color_embedding_none_lookup_lookuptablefindv2_table_handle/color_embedding/to_sparse_input/values:output:0;color_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
color_embedding/ReadVariableOpReadVariableOp'color_embedding_readvariableop_resource*
_output_shapes

:*
dtype0}
3color_embedding/color_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: |
2color_embedding/color_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
-color_embedding/color_embedding_weights/SliceSlice4color_embedding/to_sparse_input/dense_shape:output:0<color_embedding/color_embedding_weights/Slice/begin:output:0;color_embedding/color_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:w
-color_embedding/color_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
,color_embedding/color_embedding_weights/ProdProd6color_embedding/color_embedding_weights/Slice:output:06color_embedding/color_embedding_weights/Const:output:0*
T0	*
_output_shapes
: z
8color_embedding/color_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :w
5color_embedding/color_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0color_embedding/color_embedding_weights/GatherV2GatherV24color_embedding/to_sparse_input/dense_shape:output:0Acolor_embedding/color_embedding_weights/GatherV2/indices:output:0>color_embedding/color_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
.color_embedding/color_embedding_weights/Cast/xPack5color_embedding/color_embedding_weights/Prod:output:09color_embedding/color_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
5color_embedding/color_embedding_weights/SparseReshapeSparseReshape/color_embedding/to_sparse_input/indices:index:04color_embedding/to_sparse_input/dense_shape:output:07color_embedding/color_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
>color_embedding/color_embedding_weights/SparseReshape/IdentityIdentity6color_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:?????????x
6color_embedding/color_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
4color_embedding/color_embedding_weights/GreaterEqualGreaterEqualGcolor_embedding/color_embedding_weights/SparseReshape/Identity:output:0?color_embedding/color_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
-color_embedding/color_embedding_weights/WhereWhere8color_embedding/color_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
5color_embedding/color_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
/color_embedding/color_embedding_weights/ReshapeReshape5color_embedding/color_embedding_weights/Where:index:0>color_embedding/color_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:?????????y
7color_embedding/color_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2color_embedding/color_embedding_weights/GatherV2_1GatherV2Fcolor_embedding/color_embedding_weights/SparseReshape:output_indices:08color_embedding/color_embedding_weights/Reshape:output:0@color_embedding/color_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????y
7color_embedding/color_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
2color_embedding/color_embedding_weights/GatherV2_2GatherV2Gcolor_embedding/color_embedding_weights/SparseReshape/Identity:output:08color_embedding/color_embedding_weights/Reshape:output:0@color_embedding/color_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
0color_embedding/color_embedding_weights/IdentityIdentityDcolor_embedding/color_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
Acolor_embedding/color_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Ocolor_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows;color_embedding/color_embedding_weights/GatherV2_1:output:0;color_embedding/color_embedding_weights/GatherV2_2:output:09color_embedding/color_embedding_weights/Identity:output:0Jcolor_embedding/color_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
Scolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
Ucolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
Ucolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Mcolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice`color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0\color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0^color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0^color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Fcolor_embedding/color_embedding_weights/embedding_lookup_sparse/UniqueUnique_color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
Ucolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*1
_class'
%#loc:@color_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
Pcolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2&color_embedding/ReadVariableOp:value:0Jcolor_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:y:0^color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*1
_class'
%#loc:@color_embedding/ReadVariableOp*'
_output_shapes
:??????????
Ycolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityYcolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
?color_embedding/color_embedding_weights/embedding_lookup_sparseSparseSegmentMeanbcolor_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0Lcolor_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:idx:0Vcolor_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
7color_embedding/color_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
1color_embedding/color_embedding_weights/Reshape_1Reshapeecolor_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0@color_embedding/color_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
-color_embedding/color_embedding_weights/ShapeShapeHcolor_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
;color_embedding/color_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
=color_embedding/color_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=color_embedding/color_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5color_embedding/color_embedding_weights/strided_sliceStridedSlice6color_embedding/color_embedding_weights/Shape:output:0Dcolor_embedding/color_embedding_weights/strided_slice/stack:output:0Fcolor_embedding/color_embedding_weights/strided_slice/stack_1:output:0Fcolor_embedding/color_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/color_embedding/color_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
-color_embedding/color_embedding_weights/stackPack8color_embedding/color_embedding_weights/stack/0:output:0>color_embedding/color_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
,color_embedding/color_embedding_weights/TileTile:color_embedding/color_embedding_weights/Reshape_1:output:06color_embedding/color_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
2color_embedding/color_embedding_weights/zeros_like	ZerosLikeHcolor_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
'color_embedding/color_embedding_weightsSelect5color_embedding/color_embedding_weights/Tile:output:06color_embedding/color_embedding_weights/zeros_like:y:0Hcolor_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
.color_embedding/color_embedding_weights/Cast_1Cast4color_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:
5color_embedding/color_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ~
4color_embedding/color_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
/color_embedding/color_embedding_weights/Slice_1Slice2color_embedding/color_embedding_weights/Cast_1:y:0>color_embedding/color_embedding_weights/Slice_1/begin:output:0=color_embedding/color_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
/color_embedding/color_embedding_weights/Shape_1Shape0color_embedding/color_embedding_weights:output:0*
T0*
_output_shapes
:
5color_embedding/color_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
4color_embedding/color_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
/color_embedding/color_embedding_weights/Slice_2Slice8color_embedding/color_embedding_weights/Shape_1:output:0>color_embedding/color_embedding_weights/Slice_2/begin:output:0=color_embedding/color_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:u
3color_embedding/color_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.color_embedding/color_embedding_weights/concatConcatV28color_embedding/color_embedding_weights/Slice_1:output:08color_embedding/color_embedding_weights/Slice_2:output:0<color_embedding/color_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
1color_embedding/color_embedding_weights/Reshape_2Reshape0color_embedding/color_embedding_weights:output:07color_embedding/color_embedding_weights/concat:output:0*
T0*'
_output_shapes
:?????????
color_embedding/ShapeShape:color_embedding/color_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:m
#color_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%color_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%color_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
color_embedding/strided_sliceStridedSlicecolor_embedding/Shape:output:0,color_embedding/strided_slice/stack:output:0.color_embedding/strided_slice/stack_1:output:0.color_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
color_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
color_embedding/Reshape/shapePack&color_embedding/strided_slice:output:0(color_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
color_embedding/ReshapeReshape:color_embedding/color_embedding_weights/Reshape_2:output:0&color_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
concatConcatV2carat/Reshape:output:0"clarity_embedding/Reshape:output:0 color_embedding/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????^
IdentityIdentityconcat:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp0^clarity_embedding/None_Lookup/LookupTableFindV2!^clarity_embedding/ReadVariableOp.^color_embedding/None_Lookup/LookupTableFindV2^color_embedding/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : 2b
/clarity_embedding/None_Lookup/LookupTableFindV2/clarity_embedding/None_Lookup/LookupTableFindV22D
 clarity_embedding/ReadVariableOp clarity_embedding/ReadVariableOp2^
-color_embedding/None_Lookup/LookupTableFindV2-color_embedding/None_Lookup/LookupTableFindV22@
color_embedding/ReadVariableOpcolor_embedding/ReadVariableOp:S O
#
_output_shapes
:?????????
(
_user_specified_namefeatures/carat:UQ
#
_output_shapes
:?????????
*
_user_specified_namefeatures/clarity:SO
#
_output_shapes
:?????????
(
_user_specified_namefeatures/color:QM
#
_output_shapes
:?????????
&
_user_specified_namefeatures/cut:SO
#
_output_shapes
:?????????
(
_user_specified_namefeatures/depth:SO
#
_output_shapes
:?????????
(
_user_specified_namefeatures/table:OK
#
_output_shapes
:?????????
$
_user_specified_name
features/x:OK
#
_output_shapes
:?????????
$
_user_specified_name
features/y:OK
#
_output_shapes
:?????????
$
_user_specified_name
features/z:


_output_shapes
: :

_output_shapes
: 
??
?
 __inference__wrapped_model_29055	
carat
clarity		
color	
cut		
depth	
table
x
y
z\
Xsequential_2_dense_features_clarity_embedding_none_lookup_lookuptablefindv2_table_handle]
Ysequential_2_dense_features_clarity_embedding_none_lookup_lookuptablefindv2_default_value	W
Esequential_2_dense_features_clarity_embedding_readvariableop_resource:Z
Vsequential_2_dense_features_color_embedding_none_lookup_lookuptablefindv2_table_handle[
Wsequential_2_dense_features_color_embedding_none_lookup_lookuptablefindv2_default_value	U
Csequential_2_dense_features_color_embedding_readvariableop_resource:F
3sequential_2_dense_4_matmul_readvariableop_resource:	?C
4sequential_2_dense_4_biasadd_readvariableop_resource:	?F
3sequential_2_dense_5_matmul_readvariableop_resource:	?B
4sequential_2_dense_5_biasadd_readvariableop_resource:
identity??+sequential_2/dense_4/BiasAdd/ReadVariableOp?*sequential_2/dense_4/MatMul/ReadVariableOp?+sequential_2/dense_5/BiasAdd/ReadVariableOp?*sequential_2/dense_5/MatMul/ReadVariableOp?Ksequential_2/dense_features/clarity_embedding/None_Lookup/LookupTableFindV2?<sequential_2/dense_features/clarity_embedding/ReadVariableOp?Isequential_2/dense_features/color_embedding/None_Lookup/LookupTableFindV2?:sequential_2/dense_features/color_embedding/ReadVariableOp{
0sequential_2/dense_features/carat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
,sequential_2/dense_features/carat/ExpandDims
ExpandDimscarat9sequential_2/dense_features/carat/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
'sequential_2/dense_features/carat/ShapeShape5sequential_2/dense_features/carat/ExpandDims:output:0*
T0*
_output_shapes
:
5sequential_2/dense_features/carat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
7sequential_2/dense_features/carat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
7sequential_2/dense_features/carat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
/sequential_2/dense_features/carat/strided_sliceStridedSlice0sequential_2/dense_features/carat/Shape:output:0>sequential_2/dense_features/carat/strided_slice/stack:output:0@sequential_2/dense_features/carat/strided_slice/stack_1:output:0@sequential_2/dense_features/carat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1sequential_2/dense_features/carat/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
/sequential_2/dense_features/carat/Reshape/shapePack8sequential_2/dense_features/carat/strided_slice:output:0:sequential_2/dense_features/carat/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
)sequential_2/dense_features/carat/ReshapeReshape5sequential_2/dense_features/carat/ExpandDims:output:08sequential_2/dense_features/carat/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
<sequential_2/dense_features/clarity_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
8sequential_2/dense_features/clarity_embedding/ExpandDims
ExpandDimsclarityEsequential_2/dense_features/clarity_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
Lsequential_2/dense_features/clarity_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Jsequential_2/dense_features/clarity_embedding/to_sparse_input/ignore_valueCastUsequential_2/dense_features/clarity_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Fsequential_2/dense_features/clarity_embedding/to_sparse_input/NotEqualNotEqualAsequential_2/dense_features/clarity_embedding/ExpandDims:output:0Nsequential_2/dense_features/clarity_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
Esequential_2/dense_features/clarity_embedding/to_sparse_input/indicesWhereJsequential_2/dense_features/clarity_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Dsequential_2/dense_features/clarity_embedding/to_sparse_input/valuesGatherNdAsequential_2/dense_features/clarity_embedding/ExpandDims:output:0Msequential_2/dense_features/clarity_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Isequential_2/dense_features/clarity_embedding/to_sparse_input/dense_shapeShapeAsequential_2/dense_features/clarity_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
Ksequential_2/dense_features/clarity_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2Xsequential_2_dense_features_clarity_embedding_none_lookup_lookuptablefindv2_table_handleMsequential_2/dense_features/clarity_embedding/to_sparse_input/values:output:0Ysequential_2_dense_features_clarity_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
<sequential_2/dense_features/clarity_embedding/ReadVariableOpReadVariableOpEsequential_2_dense_features_clarity_embedding_readvariableop_resource*
_output_shapes

:*
dtype0?
Ssequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Rsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
Msequential_2/dense_features/clarity_embedding/clarity_embedding_weights/SliceSliceRsequential_2/dense_features/clarity_embedding/to_sparse_input/dense_shape:output:0\sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Slice/begin:output:0[sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:?
Msequential_2/dense_features/clarity_embedding/clarity_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Lsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/ProdProdVsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Slice:output:0Vsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ?
Xsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Usequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Psequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GatherV2GatherV2Rsequential_2/dense_features/clarity_embedding/to_sparse_input/dense_shape:output:0asequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GatherV2/indices:output:0^sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
Nsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Cast/xPackUsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Prod:output:0Ysequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
Usequential_2/dense_features/clarity_embedding/clarity_embedding_weights/SparseReshapeSparseReshapeMsequential_2/dense_features/clarity_embedding/to_sparse_input/indices:index:0Rsequential_2/dense_features/clarity_embedding/to_sparse_input/dense_shape:output:0Wsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
^sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/SparseReshape/IdentityIdentityTsequential_2/dense_features/clarity_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
Vsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Tsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GreaterEqualGreaterEqualgsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0_sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
Msequential_2/dense_features/clarity_embedding/clarity_embedding_weights/WhereWhereXsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
Usequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Osequential_2/dense_features/clarity_embedding/clarity_embedding_weights/ReshapeReshapeUsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Where:index:0^sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:??????????
Wsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Rsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GatherV2_1GatherV2fsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/SparseReshape:output_indices:0Xsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Reshape:output:0`sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:??????????
Wsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Rsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GatherV2_2GatherV2gsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0Xsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Reshape:output:0`sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Psequential_2/dense_features/clarity_embedding/clarity_embedding_weights/IdentityIdentitydsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
asequential_2/dense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
osequential_2/dense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows[sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GatherV2_1:output:0[sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/GatherV2_2:output:0Ysequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Identity:output:0jsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
ssequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
usequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
usequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
msequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice?sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0|sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0~sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0~sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
fsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/UniqueUniquesequential_2/dense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
usequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*O
_classE
CAloc:@sequential_2/dense_features/clarity_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
psequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Dsequential_2/dense_features/clarity_embedding/ReadVariableOp:value:0jsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:y:0~sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*O
_classE
CAloc:@sequential_2/dense_features/clarity_embedding/ReadVariableOp*'
_output_shapes
:??????????
ysequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityysequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
_sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparseSparseSegmentMean?sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0lsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:idx:0vsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
Wsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Qsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Reshape_1Reshape?sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0`sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
Msequential_2/dense_features/clarity_embedding/clarity_embedding_weights/ShapeShapehsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
[sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
]sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
]sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Usequential_2/dense_features/clarity_embedding/clarity_embedding_weights/strided_sliceStridedSliceVsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Shape:output:0dsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stack:output:0fsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stack_1:output:0fsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Osequential_2/dense_features/clarity_embedding/clarity_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
Msequential_2/dense_features/clarity_embedding/clarity_embedding_weights/stackPackXsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/stack/0:output:0^sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
Lsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/TileTileZsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Reshape_1:output:0Vsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
Rsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/zeros_like	ZerosLikehsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
Gsequential_2/dense_features/clarity_embedding/clarity_embedding_weightsSelectUsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Tile:output:0Vsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/zeros_like:y:0hsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
Nsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Cast_1CastRsequential_2/dense_features/clarity_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Usequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Tsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
Osequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Slice_1SliceRsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Cast_1:y:0^sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Slice_1/begin:output:0]sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
Osequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Shape_1ShapePsequential_2/dense_features/clarity_embedding/clarity_embedding_weights:output:0*
T0*
_output_shapes
:?
Usequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
Tsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Osequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Slice_2SliceXsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Shape_1:output:0^sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Slice_2/begin:output:0]sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:?
Ssequential_2/dense_features/clarity_embedding/clarity_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Nsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/concatConcatV2Xsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Slice_1:output:0Xsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Slice_2:output:0\sequential_2/dense_features/clarity_embedding/clarity_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Qsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Reshape_2ReshapePsequential_2/dense_features/clarity_embedding/clarity_embedding_weights:output:0Wsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/concat:output:0*
T0*'
_output_shapes
:??????????
3sequential_2/dense_features/clarity_embedding/ShapeShapeZsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:?
Asequential_2/dense_features/clarity_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Csequential_2/dense_features/clarity_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Csequential_2/dense_features/clarity_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
;sequential_2/dense_features/clarity_embedding/strided_sliceStridedSlice<sequential_2/dense_features/clarity_embedding/Shape:output:0Jsequential_2/dense_features/clarity_embedding/strided_slice/stack:output:0Lsequential_2/dense_features/clarity_embedding/strided_slice/stack_1:output:0Lsequential_2/dense_features/clarity_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
=sequential_2/dense_features/clarity_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
;sequential_2/dense_features/clarity_embedding/Reshape/shapePackDsequential_2/dense_features/clarity_embedding/strided_slice:output:0Fsequential_2/dense_features/clarity_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
5sequential_2/dense_features/clarity_embedding/ReshapeReshapeZsequential_2/dense_features/clarity_embedding/clarity_embedding_weights/Reshape_2:output:0Dsequential_2/dense_features/clarity_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
:sequential_2/dense_features/color_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
6sequential_2/dense_features/color_embedding/ExpandDims
ExpandDimscolorCsequential_2/dense_features/color_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
Jsequential_2/dense_features/color_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Hsequential_2/dense_features/color_embedding/to_sparse_input/ignore_valueCastSsequential_2/dense_features/color_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
Dsequential_2/dense_features/color_embedding/to_sparse_input/NotEqualNotEqual?sequential_2/dense_features/color_embedding/ExpandDims:output:0Lsequential_2/dense_features/color_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
Csequential_2/dense_features/color_embedding/to_sparse_input/indicesWhereHsequential_2/dense_features/color_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
Bsequential_2/dense_features/color_embedding/to_sparse_input/valuesGatherNd?sequential_2/dense_features/color_embedding/ExpandDims:output:0Ksequential_2/dense_features/color_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Gsequential_2/dense_features/color_embedding/to_sparse_input/dense_shapeShape?sequential_2/dense_features/color_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
Isequential_2/dense_features/color_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2Vsequential_2_dense_features_color_embedding_none_lookup_lookuptablefindv2_table_handleKsequential_2/dense_features/color_embedding/to_sparse_input/values:output:0Wsequential_2_dense_features_color_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
:sequential_2/dense_features/color_embedding/ReadVariableOpReadVariableOpCsequential_2_dense_features_color_embedding_readvariableop_resource*
_output_shapes

:*
dtype0?
Osequential_2/dense_features/color_embedding/color_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Nsequential_2/dense_features/color_embedding/color_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
Isequential_2/dense_features/color_embedding/color_embedding_weights/SliceSlicePsequential_2/dense_features/color_embedding/to_sparse_input/dense_shape:output:0Xsequential_2/dense_features/color_embedding/color_embedding_weights/Slice/begin:output:0Wsequential_2/dense_features/color_embedding/color_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:?
Isequential_2/dense_features/color_embedding/color_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Hsequential_2/dense_features/color_embedding/color_embedding_weights/ProdProdRsequential_2/dense_features/color_embedding/color_embedding_weights/Slice:output:0Rsequential_2/dense_features/color_embedding/color_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ?
Tsequential_2/dense_features/color_embedding/color_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Qsequential_2/dense_features/color_embedding/color_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Lsequential_2/dense_features/color_embedding/color_embedding_weights/GatherV2GatherV2Psequential_2/dense_features/color_embedding/to_sparse_input/dense_shape:output:0]sequential_2/dense_features/color_embedding/color_embedding_weights/GatherV2/indices:output:0Zsequential_2/dense_features/color_embedding/color_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
Jsequential_2/dense_features/color_embedding/color_embedding_weights/Cast/xPackQsequential_2/dense_features/color_embedding/color_embedding_weights/Prod:output:0Usequential_2/dense_features/color_embedding/color_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
Qsequential_2/dense_features/color_embedding/color_embedding_weights/SparseReshapeSparseReshapeKsequential_2/dense_features/color_embedding/to_sparse_input/indices:index:0Psequential_2/dense_features/color_embedding/to_sparse_input/dense_shape:output:0Ssequential_2/dense_features/color_embedding/color_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
Zsequential_2/dense_features/color_embedding/color_embedding_weights/SparseReshape/IdentityIdentityRsequential_2/dense_features/color_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
Rsequential_2/dense_features/color_embedding/color_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Psequential_2/dense_features/color_embedding/color_embedding_weights/GreaterEqualGreaterEqualcsequential_2/dense_features/color_embedding/color_embedding_weights/SparseReshape/Identity:output:0[sequential_2/dense_features/color_embedding/color_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
Isequential_2/dense_features/color_embedding/color_embedding_weights/WhereWhereTsequential_2/dense_features/color_embedding/color_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
Qsequential_2/dense_features/color_embedding/color_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Ksequential_2/dense_features/color_embedding/color_embedding_weights/ReshapeReshapeQsequential_2/dense_features/color_embedding/color_embedding_weights/Where:index:0Zsequential_2/dense_features/color_embedding/color_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:??????????
Ssequential_2/dense_features/color_embedding/color_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Nsequential_2/dense_features/color_embedding/color_embedding_weights/GatherV2_1GatherV2bsequential_2/dense_features/color_embedding/color_embedding_weights/SparseReshape:output_indices:0Tsequential_2/dense_features/color_embedding/color_embedding_weights/Reshape:output:0\sequential_2/dense_features/color_embedding/color_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:??????????
Ssequential_2/dense_features/color_embedding/color_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Nsequential_2/dense_features/color_embedding/color_embedding_weights/GatherV2_2GatherV2csequential_2/dense_features/color_embedding/color_embedding_weights/SparseReshape/Identity:output:0Tsequential_2/dense_features/color_embedding/color_embedding_weights/Reshape:output:0\sequential_2/dense_features/color_embedding/color_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Lsequential_2/dense_features/color_embedding/color_embedding_weights/IdentityIdentity`sequential_2/dense_features/color_embedding/color_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
]sequential_2/dense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
ksequential_2/dense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsWsequential_2/dense_features/color_embedding/color_embedding_weights/GatherV2_1:output:0Wsequential_2/dense_features/color_embedding/color_embedding_weights/GatherV2_2:output:0Usequential_2/dense_features/color_embedding/color_embedding_weights/Identity:output:0fsequential_2/dense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
osequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
qsequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
qsequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
isequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice|sequential_2/dense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0xsequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0zsequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0zsequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
bsequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/UniqueUnique{sequential_2/dense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
qsequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*M
_classC
A?loc:@sequential_2/dense_features/color_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
lsequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2Bsequential_2/dense_features/color_embedding/ReadVariableOp:value:0fsequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:y:0zsequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*M
_classC
A?loc:@sequential_2/dense_features/color_embedding/ReadVariableOp*'
_output_shapes
:??????????
usequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityusequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
[sequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparseSparseSegmentMean~sequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0hsequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:idx:0rsequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
Ssequential_2/dense_features/color_embedding/color_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Msequential_2/dense_features/color_embedding/color_embedding_weights/Reshape_1Reshape?sequential_2/dense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0\sequential_2/dense_features/color_embedding/color_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
Isequential_2/dense_features/color_embedding/color_embedding_weights/ShapeShapedsequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
Wsequential_2/dense_features/color_embedding/color_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Ysequential_2/dense_features/color_embedding/color_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Ysequential_2/dense_features/color_embedding/color_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Qsequential_2/dense_features/color_embedding/color_embedding_weights/strided_sliceStridedSliceRsequential_2/dense_features/color_embedding/color_embedding_weights/Shape:output:0`sequential_2/dense_features/color_embedding/color_embedding_weights/strided_slice/stack:output:0bsequential_2/dense_features/color_embedding/color_embedding_weights/strided_slice/stack_1:output:0bsequential_2/dense_features/color_embedding/color_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Ksequential_2/dense_features/color_embedding/color_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
Isequential_2/dense_features/color_embedding/color_embedding_weights/stackPackTsequential_2/dense_features/color_embedding/color_embedding_weights/stack/0:output:0Zsequential_2/dense_features/color_embedding/color_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
Hsequential_2/dense_features/color_embedding/color_embedding_weights/TileTileVsequential_2/dense_features/color_embedding/color_embedding_weights/Reshape_1:output:0Rsequential_2/dense_features/color_embedding/color_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
Nsequential_2/dense_features/color_embedding/color_embedding_weights/zeros_like	ZerosLikedsequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
Csequential_2/dense_features/color_embedding/color_embedding_weightsSelectQsequential_2/dense_features/color_embedding/color_embedding_weights/Tile:output:0Rsequential_2/dense_features/color_embedding/color_embedding_weights/zeros_like:y:0dsequential_2/dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
Jsequential_2/dense_features/color_embedding/color_embedding_weights/Cast_1CastPsequential_2/dense_features/color_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Qsequential_2/dense_features/color_embedding/color_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Psequential_2/dense_features/color_embedding/color_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
Ksequential_2/dense_features/color_embedding/color_embedding_weights/Slice_1SliceNsequential_2/dense_features/color_embedding/color_embedding_weights/Cast_1:y:0Zsequential_2/dense_features/color_embedding/color_embedding_weights/Slice_1/begin:output:0Ysequential_2/dense_features/color_embedding/color_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
Ksequential_2/dense_features/color_embedding/color_embedding_weights/Shape_1ShapeLsequential_2/dense_features/color_embedding/color_embedding_weights:output:0*
T0*
_output_shapes
:?
Qsequential_2/dense_features/color_embedding/color_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
Psequential_2/dense_features/color_embedding/color_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Ksequential_2/dense_features/color_embedding/color_embedding_weights/Slice_2SliceTsequential_2/dense_features/color_embedding/color_embedding_weights/Shape_1:output:0Zsequential_2/dense_features/color_embedding/color_embedding_weights/Slice_2/begin:output:0Ysequential_2/dense_features/color_embedding/color_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:?
Osequential_2/dense_features/color_embedding/color_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Jsequential_2/dense_features/color_embedding/color_embedding_weights/concatConcatV2Tsequential_2/dense_features/color_embedding/color_embedding_weights/Slice_1:output:0Tsequential_2/dense_features/color_embedding/color_embedding_weights/Slice_2:output:0Xsequential_2/dense_features/color_embedding/color_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Msequential_2/dense_features/color_embedding/color_embedding_weights/Reshape_2ReshapeLsequential_2/dense_features/color_embedding/color_embedding_weights:output:0Ssequential_2/dense_features/color_embedding/color_embedding_weights/concat:output:0*
T0*'
_output_shapes
:??????????
1sequential_2/dense_features/color_embedding/ShapeShapeVsequential_2/dense_features/color_embedding/color_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:?
?sequential_2/dense_features/color_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Asequential_2/dense_features/color_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Asequential_2/dense_features/color_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential_2/dense_features/color_embedding/strided_sliceStridedSlice:sequential_2/dense_features/color_embedding/Shape:output:0Hsequential_2/dense_features/color_embedding/strided_slice/stack:output:0Jsequential_2/dense_features/color_embedding/strided_slice/stack_1:output:0Jsequential_2/dense_features/color_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;sequential_2/dense_features/color_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
9sequential_2/dense_features/color_embedding/Reshape/shapePackBsequential_2/dense_features/color_embedding/strided_slice:output:0Dsequential_2/dense_features/color_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
3sequential_2/dense_features/color_embedding/ReshapeReshapeVsequential_2/dense_features/color_embedding/color_embedding_weights/Reshape_2:output:0Bsequential_2/dense_features/color_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????r
'sequential_2/dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
"sequential_2/dense_features/concatConcatV22sequential_2/dense_features/carat/Reshape:output:0>sequential_2/dense_features/clarity_embedding/Reshape:output:0<sequential_2/dense_features/color_embedding/Reshape:output:00sequential_2/dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
*sequential_2/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
sequential_2/dense_4/MatMulMatMul+sequential_2/dense_features/concat:output:02sequential_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
+sequential_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_2/dense_4/BiasAddBiasAdd%sequential_2/dense_4/MatMul:product:03sequential_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????{
sequential_2/dense_4/ReluRelu%sequential_2/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
*sequential_2/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
sequential_2/dense_5/MatMulMatMul'sequential_2/dense_4/Relu:activations:02sequential_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_2/dense_5/BiasAddBiasAdd%sequential_2/dense_5/MatMul:product:03sequential_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
IdentityIdentity%sequential_2/dense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^sequential_2/dense_4/BiasAdd/ReadVariableOp+^sequential_2/dense_4/MatMul/ReadVariableOp,^sequential_2/dense_5/BiasAdd/ReadVariableOp+^sequential_2/dense_5/MatMul/ReadVariableOpL^sequential_2/dense_features/clarity_embedding/None_Lookup/LookupTableFindV2=^sequential_2/dense_features/clarity_embedding/ReadVariableOpJ^sequential_2/dense_features/color_embedding/None_Lookup/LookupTableFindV2;^sequential_2/dense_features/color_embedding/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2Z
+sequential_2/dense_4/BiasAdd/ReadVariableOp+sequential_2/dense_4/BiasAdd/ReadVariableOp2X
*sequential_2/dense_4/MatMul/ReadVariableOp*sequential_2/dense_4/MatMul/ReadVariableOp2Z
+sequential_2/dense_5/BiasAdd/ReadVariableOp+sequential_2/dense_5/BiasAdd/ReadVariableOp2X
*sequential_2/dense_5/MatMul/ReadVariableOp*sequential_2/dense_5/MatMul/ReadVariableOp2?
Ksequential_2/dense_features/clarity_embedding/None_Lookup/LookupTableFindV2Ksequential_2/dense_features/clarity_embedding/None_Lookup/LookupTableFindV22|
<sequential_2/dense_features/clarity_embedding/ReadVariableOp<sequential_2/dense_features/clarity_embedding/ReadVariableOp2?
Isequential_2/dense_features/color_embedding/None_Lookup/LookupTableFindV2Isequential_2/dense_features/color_embedding/None_Lookup/LookupTableFindV22x
:sequential_2/dense_features/color_embedding/ReadVariableOp:sequential_2/dense_features/color_embedding/ReadVariableOp:J F
#
_output_shapes
:?????????

_user_specified_namecarat:LH
#
_output_shapes
:?????????
!
_user_specified_name	clarity:JF
#
_output_shapes
:?????????

_user_specified_namecolor:HD
#
_output_shapes
:?????????

_user_specified_namecut:JF
#
_output_shapes
:?????????

_user_specified_namedepth:JF
#
_output_shapes
:?????????

_user_specified_nametable:FB
#
_output_shapes
:?????????

_user_specified_namex:FB
#
_output_shapes
:?????????

_user_specified_namey:FB
#
_output_shapes
:?????????

_user_specified_namez:


_output_shapes
: :

_output_shapes
: 
??
?

G__inference_sequential_2_layer_call_and_return_conditional_losses_30067
inputs_carat
inputs_clarity	
inputs_color	

inputs_cut	
inputs_depth
inputs_table
inputs_x
inputs_y
inputs_zO
Kdense_features_clarity_embedding_none_lookup_lookuptablefindv2_table_handleP
Ldense_features_clarity_embedding_none_lookup_lookuptablefindv2_default_value	J
8dense_features_clarity_embedding_readvariableop_resource:M
Idense_features_color_embedding_none_lookup_lookuptablefindv2_table_handleN
Jdense_features_color_embedding_none_lookup_lookuptablefindv2_default_value	H
6dense_features_color_embedding_readvariableop_resource:9
&dense_4_matmul_readvariableop_resource:	?6
'dense_4_biasadd_readvariableop_resource:	?9
&dense_5_matmul_readvariableop_resource:	?5
'dense_5_biasadd_readvariableop_resource:
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?>dense_features/clarity_embedding/None_Lookup/LookupTableFindV2?/dense_features/clarity_embedding/ReadVariableOp?<dense_features/color_embedding/None_Lookup/LookupTableFindV2?-dense_features/color_embedding/ReadVariableOpn
#dense_features/carat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
dense_features/carat/ExpandDims
ExpandDimsinputs_carat,dense_features/carat/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????r
dense_features/carat/ShapeShape(dense_features/carat/ExpandDims:output:0*
T0*
_output_shapes
:r
(dense_features/carat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*dense_features/carat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*dense_features/carat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"dense_features/carat/strided_sliceStridedSlice#dense_features/carat/Shape:output:01dense_features/carat/strided_slice/stack:output:03dense_features/carat/strided_slice/stack_1:output:03dense_features/carat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$dense_features/carat/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
"dense_features/carat/Reshape/shapePack+dense_features/carat/strided_slice:output:0-dense_features/carat/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
dense_features/carat/ReshapeReshape(dense_features/carat/ExpandDims:output:0+dense_features/carat/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????z
/dense_features/clarity_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
+dense_features/clarity_embedding/ExpandDims
ExpandDimsinputs_clarity8dense_features/clarity_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
?dense_features/clarity_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
=dense_features/clarity_embedding/to_sparse_input/ignore_valueCastHdense_features/clarity_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
9dense_features/clarity_embedding/to_sparse_input/NotEqualNotEqual4dense_features/clarity_embedding/ExpandDims:output:0Adense_features/clarity_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
8dense_features/clarity_embedding/to_sparse_input/indicesWhere=dense_features/clarity_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
7dense_features/clarity_embedding/to_sparse_input/valuesGatherNd4dense_features/clarity_embedding/ExpandDims:output:0@dense_features/clarity_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
<dense_features/clarity_embedding/to_sparse_input/dense_shapeShape4dense_features/clarity_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
>dense_features/clarity_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2Kdense_features_clarity_embedding_none_lookup_lookuptablefindv2_table_handle@dense_features/clarity_embedding/to_sparse_input/values:output:0Ldense_features_clarity_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
/dense_features/clarity_embedding/ReadVariableOpReadVariableOp8dense_features_clarity_embedding_readvariableop_resource*
_output_shapes

:*
dtype0?
Fdense_features/clarity_embedding/clarity_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Edense_features/clarity_embedding/clarity_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
@dense_features/clarity_embedding/clarity_embedding_weights/SliceSliceEdense_features/clarity_embedding/to_sparse_input/dense_shape:output:0Odense_features/clarity_embedding/clarity_embedding_weights/Slice/begin:output:0Ndense_features/clarity_embedding/clarity_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:?
@dense_features/clarity_embedding/clarity_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
?dense_features/clarity_embedding/clarity_embedding_weights/ProdProdIdense_features/clarity_embedding/clarity_embedding_weights/Slice:output:0Idense_features/clarity_embedding/clarity_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ?
Kdense_features/clarity_embedding/clarity_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Hdense_features/clarity_embedding/clarity_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Cdense_features/clarity_embedding/clarity_embedding_weights/GatherV2GatherV2Edense_features/clarity_embedding/to_sparse_input/dense_shape:output:0Tdense_features/clarity_embedding/clarity_embedding_weights/GatherV2/indices:output:0Qdense_features/clarity_embedding/clarity_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
Adense_features/clarity_embedding/clarity_embedding_weights/Cast/xPackHdense_features/clarity_embedding/clarity_embedding_weights/Prod:output:0Ldense_features/clarity_embedding/clarity_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
Hdense_features/clarity_embedding/clarity_embedding_weights/SparseReshapeSparseReshape@dense_features/clarity_embedding/to_sparse_input/indices:index:0Edense_features/clarity_embedding/to_sparse_input/dense_shape:output:0Jdense_features/clarity_embedding/clarity_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
Qdense_features/clarity_embedding/clarity_embedding_weights/SparseReshape/IdentityIdentityGdense_features/clarity_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
Idense_features/clarity_embedding/clarity_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Gdense_features/clarity_embedding/clarity_embedding_weights/GreaterEqualGreaterEqualZdense_features/clarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0Rdense_features/clarity_embedding/clarity_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
@dense_features/clarity_embedding/clarity_embedding_weights/WhereWhereKdense_features/clarity_embedding/clarity_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
Hdense_features/clarity_embedding/clarity_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Bdense_features/clarity_embedding/clarity_embedding_weights/ReshapeReshapeHdense_features/clarity_embedding/clarity_embedding_weights/Where:index:0Qdense_features/clarity_embedding/clarity_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:??????????
Jdense_features/clarity_embedding/clarity_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Edense_features/clarity_embedding/clarity_embedding_weights/GatherV2_1GatherV2Ydense_features/clarity_embedding/clarity_embedding_weights/SparseReshape:output_indices:0Kdense_features/clarity_embedding/clarity_embedding_weights/Reshape:output:0Sdense_features/clarity_embedding/clarity_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:??????????
Jdense_features/clarity_embedding/clarity_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Edense_features/clarity_embedding/clarity_embedding_weights/GatherV2_2GatherV2Zdense_features/clarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0Kdense_features/clarity_embedding/clarity_embedding_weights/Reshape:output:0Sdense_features/clarity_embedding/clarity_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Cdense_features/clarity_embedding/clarity_embedding_weights/IdentityIdentityWdense_features/clarity_embedding/clarity_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
Tdense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
bdense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsNdense_features/clarity_embedding/clarity_embedding_weights/GatherV2_1:output:0Ndense_features/clarity_embedding/clarity_embedding_weights/GatherV2_2:output:0Ldense_features/clarity_embedding/clarity_embedding_weights/Identity:output:0]dense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
fdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
hdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
hdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
`dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicesdense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0odense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0qdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0qdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Ydense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/UniqueUniquerdense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
hdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*B
_class8
64loc:@dense_features/clarity_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
cdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV27dense_features/clarity_embedding/ReadVariableOp:value:0]dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:y:0qdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*B
_class8
64loc:@dense_features/clarity_embedding/ReadVariableOp*'
_output_shapes
:??????????
ldense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityldense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
Rdense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparseSparseSegmentMeanudense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0_dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:idx:0idense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
Jdense_features/clarity_embedding/clarity_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Ddense_features/clarity_embedding/clarity_embedding_weights/Reshape_1Reshapexdense_features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Sdense_features/clarity_embedding/clarity_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
@dense_features/clarity_embedding/clarity_embedding_weights/ShapeShape[dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
Ndense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Pdense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Pdense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Hdense_features/clarity_embedding/clarity_embedding_weights/strided_sliceStridedSliceIdense_features/clarity_embedding/clarity_embedding_weights/Shape:output:0Wdense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stack:output:0Ydense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stack_1:output:0Ydense_features/clarity_embedding/clarity_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Bdense_features/clarity_embedding/clarity_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
@dense_features/clarity_embedding/clarity_embedding_weights/stackPackKdense_features/clarity_embedding/clarity_embedding_weights/stack/0:output:0Qdense_features/clarity_embedding/clarity_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
?dense_features/clarity_embedding/clarity_embedding_weights/TileTileMdense_features/clarity_embedding/clarity_embedding_weights/Reshape_1:output:0Idense_features/clarity_embedding/clarity_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
Edense_features/clarity_embedding/clarity_embedding_weights/zeros_like	ZerosLike[dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
:dense_features/clarity_embedding/clarity_embedding_weightsSelectHdense_features/clarity_embedding/clarity_embedding_weights/Tile:output:0Idense_features/clarity_embedding/clarity_embedding_weights/zeros_like:y:0[dense_features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
Adense_features/clarity_embedding/clarity_embedding_weights/Cast_1CastEdense_features/clarity_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Hdense_features/clarity_embedding/clarity_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Gdense_features/clarity_embedding/clarity_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
Bdense_features/clarity_embedding/clarity_embedding_weights/Slice_1SliceEdense_features/clarity_embedding/clarity_embedding_weights/Cast_1:y:0Qdense_features/clarity_embedding/clarity_embedding_weights/Slice_1/begin:output:0Pdense_features/clarity_embedding/clarity_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
Bdense_features/clarity_embedding/clarity_embedding_weights/Shape_1ShapeCdense_features/clarity_embedding/clarity_embedding_weights:output:0*
T0*
_output_shapes
:?
Hdense_features/clarity_embedding/clarity_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
Gdense_features/clarity_embedding/clarity_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Bdense_features/clarity_embedding/clarity_embedding_weights/Slice_2SliceKdense_features/clarity_embedding/clarity_embedding_weights/Shape_1:output:0Qdense_features/clarity_embedding/clarity_embedding_weights/Slice_2/begin:output:0Pdense_features/clarity_embedding/clarity_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:?
Fdense_features/clarity_embedding/clarity_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Adense_features/clarity_embedding/clarity_embedding_weights/concatConcatV2Kdense_features/clarity_embedding/clarity_embedding_weights/Slice_1:output:0Kdense_features/clarity_embedding/clarity_embedding_weights/Slice_2:output:0Odense_features/clarity_embedding/clarity_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Ddense_features/clarity_embedding/clarity_embedding_weights/Reshape_2ReshapeCdense_features/clarity_embedding/clarity_embedding_weights:output:0Jdense_features/clarity_embedding/clarity_embedding_weights/concat:output:0*
T0*'
_output_shapes
:??????????
&dense_features/clarity_embedding/ShapeShapeMdense_features/clarity_embedding/clarity_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:~
4dense_features/clarity_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6dense_features/clarity_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6dense_features/clarity_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.dense_features/clarity_embedding/strided_sliceStridedSlice/dense_features/clarity_embedding/Shape:output:0=dense_features/clarity_embedding/strided_slice/stack:output:0?dense_features/clarity_embedding/strided_slice/stack_1:output:0?dense_features/clarity_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0dense_features/clarity_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
.dense_features/clarity_embedding/Reshape/shapePack7dense_features/clarity_embedding/strided_slice:output:09dense_features/clarity_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
(dense_features/clarity_embedding/ReshapeReshapeMdense_features/clarity_embedding/clarity_embedding_weights/Reshape_2:output:07dense_features/clarity_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????x
-dense_features/color_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
)dense_features/color_embedding/ExpandDims
ExpandDimsinputs_color6dense_features/color_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
=dense_features/color_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
;dense_features/color_embedding/to_sparse_input/ignore_valueCastFdense_features/color_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
7dense_features/color_embedding/to_sparse_input/NotEqualNotEqual2dense_features/color_embedding/ExpandDims:output:0?dense_features/color_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
6dense_features/color_embedding/to_sparse_input/indicesWhere;dense_features/color_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
5dense_features/color_embedding/to_sparse_input/valuesGatherNd2dense_features/color_embedding/ExpandDims:output:0>dense_features/color_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
:dense_features/color_embedding/to_sparse_input/dense_shapeShape2dense_features/color_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
<dense_features/color_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2Idense_features_color_embedding_none_lookup_lookuptablefindv2_table_handle>dense_features/color_embedding/to_sparse_input/values:output:0Jdense_features_color_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
-dense_features/color_embedding/ReadVariableOpReadVariableOp6dense_features_color_embedding_readvariableop_resource*
_output_shapes

:*
dtype0?
Bdense_features/color_embedding/color_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Adense_features/color_embedding/color_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
<dense_features/color_embedding/color_embedding_weights/SliceSliceCdense_features/color_embedding/to_sparse_input/dense_shape:output:0Kdense_features/color_embedding/color_embedding_weights/Slice/begin:output:0Jdense_features/color_embedding/color_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:?
<dense_features/color_embedding/color_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
;dense_features/color_embedding/color_embedding_weights/ProdProdEdense_features/color_embedding/color_embedding_weights/Slice:output:0Edense_features/color_embedding/color_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ?
Gdense_features/color_embedding/color_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Ddense_features/color_embedding/color_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
?dense_features/color_embedding/color_embedding_weights/GatherV2GatherV2Cdense_features/color_embedding/to_sparse_input/dense_shape:output:0Pdense_features/color_embedding/color_embedding_weights/GatherV2/indices:output:0Mdense_features/color_embedding/color_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
=dense_features/color_embedding/color_embedding_weights/Cast/xPackDdense_features/color_embedding/color_embedding_weights/Prod:output:0Hdense_features/color_embedding/color_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
Ddense_features/color_embedding/color_embedding_weights/SparseReshapeSparseReshape>dense_features/color_embedding/to_sparse_input/indices:index:0Cdense_features/color_embedding/to_sparse_input/dense_shape:output:0Fdense_features/color_embedding/color_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
Mdense_features/color_embedding/color_embedding_weights/SparseReshape/IdentityIdentityEdense_features/color_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
Edense_features/color_embedding/color_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Cdense_features/color_embedding/color_embedding_weights/GreaterEqualGreaterEqualVdense_features/color_embedding/color_embedding_weights/SparseReshape/Identity:output:0Ndense_features/color_embedding/color_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
<dense_features/color_embedding/color_embedding_weights/WhereWhereGdense_features/color_embedding/color_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
Ddense_features/color_embedding/color_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
>dense_features/color_embedding/color_embedding_weights/ReshapeReshapeDdense_features/color_embedding/color_embedding_weights/Where:index:0Mdense_features/color_embedding/color_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:??????????
Fdense_features/color_embedding/color_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Adense_features/color_embedding/color_embedding_weights/GatherV2_1GatherV2Udense_features/color_embedding/color_embedding_weights/SparseReshape:output_indices:0Gdense_features/color_embedding/color_embedding_weights/Reshape:output:0Odense_features/color_embedding/color_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:??????????
Fdense_features/color_embedding/color_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Adense_features/color_embedding/color_embedding_weights/GatherV2_2GatherV2Vdense_features/color_embedding/color_embedding_weights/SparseReshape/Identity:output:0Gdense_features/color_embedding/color_embedding_weights/Reshape:output:0Odense_features/color_embedding/color_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
?dense_features/color_embedding/color_embedding_weights/IdentityIdentitySdense_features/color_embedding/color_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
Pdense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
^dense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsJdense_features/color_embedding/color_embedding_weights/GatherV2_1:output:0Jdense_features/color_embedding/color_embedding_weights/GatherV2_2:output:0Hdense_features/color_embedding/color_embedding_weights/Identity:output:0Ydense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
bdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
ddense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
ddense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
\dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceodense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0kdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0mdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0mdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Udense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/UniqueUniquendense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
ddense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*@
_class6
42loc:@dense_features/color_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
_dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV25dense_features/color_embedding/ReadVariableOp:value:0Ydense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:y:0mdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*@
_class6
42loc:@dense_features/color_embedding/ReadVariableOp*'
_output_shapes
:??????????
hdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityhdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
Ndense_features/color_embedding/color_embedding_weights/embedding_lookup_sparseSparseSegmentMeanqdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0[dense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:idx:0edense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
Fdense_features/color_embedding/color_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
@dense_features/color_embedding/color_embedding_weights/Reshape_1Reshapetdense_features/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Odense_features/color_embedding/color_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
<dense_features/color_embedding/color_embedding_weights/ShapeShapeWdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
Jdense_features/color_embedding/color_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Ldense_features/color_embedding/color_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Ldense_features/color_embedding/color_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Ddense_features/color_embedding/color_embedding_weights/strided_sliceStridedSliceEdense_features/color_embedding/color_embedding_weights/Shape:output:0Sdense_features/color_embedding/color_embedding_weights/strided_slice/stack:output:0Udense_features/color_embedding/color_embedding_weights/strided_slice/stack_1:output:0Udense_features/color_embedding/color_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
>dense_features/color_embedding/color_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
<dense_features/color_embedding/color_embedding_weights/stackPackGdense_features/color_embedding/color_embedding_weights/stack/0:output:0Mdense_features/color_embedding/color_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
;dense_features/color_embedding/color_embedding_weights/TileTileIdense_features/color_embedding/color_embedding_weights/Reshape_1:output:0Edense_features/color_embedding/color_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
Adense_features/color_embedding/color_embedding_weights/zeros_like	ZerosLikeWdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
6dense_features/color_embedding/color_embedding_weightsSelectDdense_features/color_embedding/color_embedding_weights/Tile:output:0Edense_features/color_embedding/color_embedding_weights/zeros_like:y:0Wdense_features/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
=dense_features/color_embedding/color_embedding_weights/Cast_1CastCdense_features/color_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Ddense_features/color_embedding/color_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Cdense_features/color_embedding/color_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
>dense_features/color_embedding/color_embedding_weights/Slice_1SliceAdense_features/color_embedding/color_embedding_weights/Cast_1:y:0Mdense_features/color_embedding/color_embedding_weights/Slice_1/begin:output:0Ldense_features/color_embedding/color_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
>dense_features/color_embedding/color_embedding_weights/Shape_1Shape?dense_features/color_embedding/color_embedding_weights:output:0*
T0*
_output_shapes
:?
Ddense_features/color_embedding/color_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
Cdense_features/color_embedding/color_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
>dense_features/color_embedding/color_embedding_weights/Slice_2SliceGdense_features/color_embedding/color_embedding_weights/Shape_1:output:0Mdense_features/color_embedding/color_embedding_weights/Slice_2/begin:output:0Ldense_features/color_embedding/color_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:?
Bdense_features/color_embedding/color_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
=dense_features/color_embedding/color_embedding_weights/concatConcatV2Gdense_features/color_embedding/color_embedding_weights/Slice_1:output:0Gdense_features/color_embedding/color_embedding_weights/Slice_2:output:0Kdense_features/color_embedding/color_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
@dense_features/color_embedding/color_embedding_weights/Reshape_2Reshape?dense_features/color_embedding/color_embedding_weights:output:0Fdense_features/color_embedding/color_embedding_weights/concat:output:0*
T0*'
_output_shapes
:??????????
$dense_features/color_embedding/ShapeShapeIdense_features/color_embedding/color_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:|
2dense_features/color_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4dense_features/color_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4dense_features/color_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,dense_features/color_embedding/strided_sliceStridedSlice-dense_features/color_embedding/Shape:output:0;dense_features/color_embedding/strided_slice/stack:output:0=dense_features/color_embedding/strided_slice/stack_1:output:0=dense_features/color_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.dense_features/color_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
,dense_features/color_embedding/Reshape/shapePack5dense_features/color_embedding/strided_slice:output:07dense_features/color_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
&dense_features/color_embedding/ReshapeReshapeIdense_features/color_embedding/color_embedding_weights/Reshape_2:output:05dense_features/color_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????e
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
dense_features/concatConcatV2%dense_features/carat/Reshape:output:01dense_features/clarity_embedding/Reshape:output:0/dense_features/color_embedding/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_4/MatMulMatMuldense_features/concat:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????a
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp?^dense_features/clarity_embedding/None_Lookup/LookupTableFindV20^dense_features/clarity_embedding/ReadVariableOp=^dense_features/color_embedding/None_Lookup/LookupTableFindV2.^dense_features/color_embedding/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2?
>dense_features/clarity_embedding/None_Lookup/LookupTableFindV2>dense_features/clarity_embedding/None_Lookup/LookupTableFindV22b
/dense_features/clarity_embedding/ReadVariableOp/dense_features/clarity_embedding/ReadVariableOp2|
<dense_features/color_embedding/None_Lookup/LookupTableFindV2<dense_features/color_embedding/None_Lookup/LookupTableFindV22^
-dense_features/color_embedding/ReadVariableOp-dense_features/color_embedding/ReadVariableOp:Q M
#
_output_shapes
:?????????
&
_user_specified_nameinputs/carat:SO
#
_output_shapes
:?????????
(
_user_specified_nameinputs/clarity:QM
#
_output_shapes
:?????????
&
_user_specified_nameinputs/color:OK
#
_output_shapes
:?????????
$
_user_specified_name
inputs/cut:QM
#
_output_shapes
:?????????
&
_user_specified_nameinputs/depth:QM
#
_output_shapes
:?????????
&
_user_specified_nameinputs/table:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/x:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/y:MI
#
_output_shapes
:?????????
"
_user_specified_name
inputs/z:


_output_shapes
: :

_output_shapes
: 
?
?
,__inference_sequential_2_layer_call_fn_29335	
carat
clarity		
color	
cut		
depth	
table
x
y
z
unknown
	unknown_0	
	unknown_1:
	unknown_2
	unknown_3	
	unknown_4:
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallcaratclaritycolorcutdepthtablexyzunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_29312o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
#
_output_shapes
:?????????

_user_specified_namecarat:LH
#
_output_shapes
:?????????
!
_user_specified_name	clarity:JF
#
_output_shapes
:?????????

_user_specified_namecolor:HD
#
_output_shapes
:?????????

_user_specified_namecut:JF
#
_output_shapes
:?????????

_user_specified_namedepth:JF
#
_output_shapes
:?????????

_user_specified_nametable:FB
#
_output_shapes
:?????????

_user_specified_namex:FB
#
_output_shapes
:?????????

_user_specified_namey:FB
#
_output_shapes
:?????????

_user_specified_namez:


_output_shapes
: :

_output_shapes
: 
?
?
__inference_<lambda>_308142
.table_init557_lookuptableimportv2_table_handle*
&table_init557_lookuptableimportv2_keys	,
(table_init557_lookuptableimportv2_values	
identity??!table_init557/LookupTableImportV2?
!table_init557/LookupTableImportV2LookupTableImportV2.table_init557_lookuptableimportv2_table_handle&table_init557_lookuptableimportv2_keys(table_init557_lookuptableimportv2_values*	
Tin0	*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: j
NoOpNoOp"^table_init557/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2F
!table_init557/LookupTableImportV2!table_init557/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:"?L
saver_filename:0StatefulPartitionedCall_3:0StatefulPartitionedCall_48"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
3
carat*
serving_default_carat:0?????????
7
clarity,
serving_default_clarity:0	?????????
3
color*
serving_default_color:0	?????????
/
cut(
serving_default_cut:0	?????????
3
depth*
serving_default_depth:0?????????
3
table*
serving_default_table:0?????????
+
x&
serving_default_x:0?????????
+
y&
serving_default_y:0?????????
+
z&
serving_default_z:0?????????>
output_12
StatefulPartitionedCall_2:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
_build_input_shape
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
?
_feature_columns

_resources
'#clarity_embedding/embedding_weights
%!color_embedding/embedding_weights
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
?
(iter

)beta_1

*beta_2
	+decay
,learning_ratemXmYmZm[ m\!m]v^v_v`va vb!vc"
	optimizer
 "
trackable_dict_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
-non_trainable_variables

.layers
/metrics
0layer_regularization_losses
1layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_sequential_2_layer_call_fn_29335
,__inference_sequential_2_layer_call_fn_29835
,__inference_sequential_2_layer_call_fn_29868
,__inference_sequential_2_layer_call_fn_29726?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_2_layer_call_and_return_conditional_losses_30067
G__inference_sequential_2_layer_call_and_return_conditional_losses_30266
G__inference_sequential_2_layer_call_and_return_conditional_losses_29761
G__inference_sequential_2_layer_call_and_return_conditional_losses_29796?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_29055caratclaritycolorcutdepthtablexyz	"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
2serving_default"
signature_map
 "
trackable_list_wrapper
6
3clarity
	4color"
_generic_user_object
Q:O2?sequential_1/dense_features/clarity_embedding/embedding_weights
O:M2=sequential_1/dense_features/color_embedding/embedding_weights
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_dense_features_layer_call_fn_30326
.__inference_dense_features_layer_call_fn_30351?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_dense_features_layer_call_and_return_conditional_losses_30537
I__inference_dense_features_layer_call_and_return_conditional_losses_30723?
???
FullArgSpecE
args=?:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
.:,	?2sequential_2/dense_4/kernel
(:&?2sequential_2/dense_4/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_dense_4_layer_call_fn_30732?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_4_layer_call_and_return_conditional_losses_30743?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.:,	?2sequential_2/dense_5/kernel
':%2sequential_2/dense_5/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_dense_5_layer_call_fn_30752?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_5_layer_call_and_return_conditional_losses_30762?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
#__inference_signature_wrapper_30301caratclaritycolorcutdepthtablexyz"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
2
Fclarity_lookup"
_generic_user_object
0
Gcolor_lookup"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Htotal
	Icount
J	variables
K	keras_api"
_tf_keras_metric
N
	Ltotal
	Mcount
N	variables
O	keras_api"
_tf_keras_metric
j
P_initializer
Q_create_resource
R_initialize
S_destroy_resourceR jCustom.StaticHashTable
j
T_initializer
U_create_resource
V_initialize
W_destroy_resourceR jCustom.StaticHashTable
:  (2total
:  (2count
.
H0
I1"
trackable_list_wrapper
-
J	variables"
_generic_user_object
:  (2total
:  (2count
.
L0
M1"
trackable_list_wrapper
-
N	variables"
_generic_user_object
"
_generic_user_object
?2?
__inference__creator_30767?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_30775?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_30780?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
"
_generic_user_object
?2?
__inference__creator_30785?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_30793?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_30798?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
V:T2FAdam/sequential_1/dense_features/clarity_embedding/embedding_weights/m
T:R2DAdam/sequential_1/dense_features/color_embedding/embedding_weights/m
3:1	?2"Adam/sequential_2/dense_4/kernel/m
-:+?2 Adam/sequential_2/dense_4/bias/m
3:1	?2"Adam/sequential_2/dense_5/kernel/m
,:*2 Adam/sequential_2/dense_5/bias/m
V:T2FAdam/sequential_1/dense_features/clarity_embedding/embedding_weights/v
T:R2DAdam/sequential_1/dense_features/color_embedding/embedding_weights/v
3:1	?2"Adam/sequential_2/dense_4/kernel/v
-:+?2 Adam/sequential_2/dense_4/bias/v
3:1	?2"Adam/sequential_2/dense_5/kernel/v
,:*2 Adam/sequential_2/dense_5/bias/v
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_56
__inference__creator_30767?

? 
? "? 6
__inference__creator_30785?

? 
? "? 8
__inference__destroyer_30780?

? 
? "? 8
__inference__destroyer_30798?

? 
? "? ?
__inference__initializer_30775Ffg?

? 
? "? ?
__inference__initializer_30793Ghi?

? 
? "? ?
 __inference__wrapped_model_29055?
FdGe !???
???
???
$
carat?
carat?????????
(
clarity?
clarity?????????	
$
color?
color?????????	
 
cut?
cut?????????	
$
depth?
depth?????????
$
table?
table?????????

x?
x?????????

y?
y?????????

z?
z?????????
? "3?0
.
output_1"?
output_1??????????
B__inference_dense_4_layer_call_and_return_conditional_losses_30743]/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? {
'__inference_dense_4_layer_call_fn_30732P/?,
%?"
 ?
inputs?????????
? "????????????
B__inference_dense_5_layer_call_and_return_conditional_losses_30762] !0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_5_layer_call_fn_30752P !0?-
&?#
!?
inputs??????????
? "???????????
I__inference_dense_features_layer_call_and_return_conditional_losses_30537?FdGe???
???
???
-
carat$?!
features/carat?????????
1
clarity&?#
features/clarity?????????	
-
color$?!
features/color?????????	
)
cut"?
features/cut?????????	
-
depth$?!
features/depth?????????
-
table$?!
features/table?????????
%
x ?

features/x?????????
%
y ?

features/y?????????
%
z ?

features/z?????????

 
p 
? "%?"
?
0?????????
? ?
I__inference_dense_features_layer_call_and_return_conditional_losses_30723?FdGe???
???
???
-
carat$?!
features/carat?????????
1
clarity&?#
features/clarity?????????	
-
color$?!
features/color?????????	
)
cut"?
features/cut?????????	
-
depth$?!
features/depth?????????
-
table$?!
features/table?????????
%
x ?

features/x?????????
%
y ?

features/y?????????
%
z ?

features/z?????????

 
p
? "%?"
?
0?????????
? ?
.__inference_dense_features_layer_call_fn_30326?FdGe???
???
???
-
carat$?!
features/carat?????????
1
clarity&?#
features/clarity?????????	
-
color$?!
features/color?????????	
)
cut"?
features/cut?????????	
-
depth$?!
features/depth?????????
-
table$?!
features/table?????????
%
x ?

features/x?????????
%
y ?

features/y?????????
%
z ?

features/z?????????

 
p 
? "???????????
.__inference_dense_features_layer_call_fn_30351?FdGe???
???
???
-
carat$?!
features/carat?????????
1
clarity&?#
features/clarity?????????	
-
color$?!
features/color?????????	
)
cut"?
features/cut?????????	
-
depth$?!
features/depth?????????
-
table$?!
features/table?????????
%
x ?

features/x?????????
%
y ?

features/y?????????
%
z ?

features/z?????????

 
p
? "???????????
G__inference_sequential_2_layer_call_and_return_conditional_losses_29761?
FdGe !???
???
???
$
carat?
carat?????????
(
clarity?
clarity?????????	
$
color?
color?????????	
 
cut?
cut?????????	
$
depth?
depth?????????
$
table?
table?????????

x?
x?????????

y?
y?????????

z?
z?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_29796?
FdGe !???
???
???
$
carat?
carat?????????
(
clarity?
clarity?????????	
$
color?
color?????????	
 
cut?
cut?????????	
$
depth?
depth?????????
$
table?
table?????????

x?
x?????????

y?
y?????????

z?
z?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_30067?
FdGe !???
???
???
+
carat"?
inputs/carat?????????
/
clarity$?!
inputs/clarity?????????	
+
color"?
inputs/color?????????	
'
cut ?

inputs/cut?????????	
+
depth"?
inputs/depth?????????
+
table"?
inputs/table?????????
#
x?
inputs/x?????????
#
y?
inputs/y?????????
#
z?
inputs/z?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_sequential_2_layer_call_and_return_conditional_losses_30266?
FdGe !???
???
???
+
carat"?
inputs/carat?????????
/
clarity$?!
inputs/clarity?????????	
+
color"?
inputs/color?????????	
'
cut ?

inputs/cut?????????	
+
depth"?
inputs/depth?????????
+
table"?
inputs/table?????????
#
x?
inputs/x?????????
#
y?
inputs/y?????????
#
z?
inputs/z?????????
p

 
? "%?"
?
0?????????
? ?
,__inference_sequential_2_layer_call_fn_29335?
FdGe !???
???
???
$
carat?
carat?????????
(
clarity?
clarity?????????	
$
color?
color?????????	
 
cut?
cut?????????	
$
depth?
depth?????????
$
table?
table?????????

x?
x?????????

y?
y?????????

z?
z?????????
p 

 
? "???????????
,__inference_sequential_2_layer_call_fn_29726?
FdGe !???
???
???
$
carat?
carat?????????
(
clarity?
clarity?????????	
$
color?
color?????????	
 
cut?
cut?????????	
$
depth?
depth?????????
$
table?
table?????????

x?
x?????????

y?
y?????????

z?
z?????????
p

 
? "???????????
,__inference_sequential_2_layer_call_fn_29835?
FdGe !???
???
???
+
carat"?
inputs/carat?????????
/
clarity$?!
inputs/clarity?????????	
+
color"?
inputs/color?????????	
'
cut ?

inputs/cut?????????	
+
depth"?
inputs/depth?????????
+
table"?
inputs/table?????????
#
x?
inputs/x?????????
#
y?
inputs/y?????????
#
z?
inputs/z?????????
p 

 
? "???????????
,__inference_sequential_2_layer_call_fn_29868?
FdGe !???
???
???
+
carat"?
inputs/carat?????????
/
clarity$?!
inputs/clarity?????????	
+
color"?
inputs/color?????????	
'
cut ?

inputs/cut?????????	
+
depth"?
inputs/depth?????????
+
table"?
inputs/table?????????
#
x?
inputs/x?????????
#
y?
inputs/y?????????
#
z?
inputs/z?????????
p

 
? "???????????
#__inference_signature_wrapper_30301?
FdGe !???
? 
???
$
carat?
carat?????????
(
clarity?
clarity?????????	
$
color?
color?????????	
 
cut?
cut?????????	
$
depth?
depth?????????
$
table?
table?????????

x?
x?????????

y?
y?????????

z?
z?????????"3?0
.
output_1"?
output_1?????????