??
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
0
Sigmoid
x"T
y"T"
Ttype:

2
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
9sequential_2/features/clarity_embedding/embedding_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*J
shared_name;9sequential_2/features/clarity_embedding/embedding_weights
?
Msequential_2/features/clarity_embedding/embedding_weights/Read/ReadVariableOpReadVariableOp9sequential_2/features/clarity_embedding/embedding_weights*
_output_shapes

:*
dtype0
?
7sequential_2/features/color_embedding/embedding_weightsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*H
shared_name97sequential_2/features/color_embedding/embedding_weights
?
Ksequential_2/features/color_embedding/embedding_weights/Read/ReadVariableOpReadVariableOp7sequential_2/features/color_embedding/embedding_weights*
_output_shapes

:*
dtype0
?
sequential_2/middlerelu/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name sequential_2/middlerelu/kernel
?
2sequential_2/middlerelu/kernel/Read/ReadVariableOpReadVariableOpsequential_2/middlerelu/kernel*
_output_shapes
:	?*
dtype0
?
sequential_2/middlerelu/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namesequential_2/middlerelu/bias
?
0sequential_2/middlerelu/bias/Read/ReadVariableOpReadVariableOpsequential_2/middlerelu/bias*
_output_shapes	
:?*
dtype0
?
sequential_2/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*,
shared_namesequential_2/dense_2/kernel
?
/sequential_2/dense_2/kernel/Read/ReadVariableOpReadVariableOpsequential_2/dense_2/kernel*
_output_shapes
:	?*
dtype0
?
sequential_2/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namesequential_2/dense_2/bias
?
-sequential_2/dense_2/bias/Read/ReadVariableOpReadVariableOpsequential_2/dense_2/bias*
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
n

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name360947*
value_dtype0	
p
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name361039*
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
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
x
true_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:d*!
shared_nametrue_positives_2
q
$true_positives_2/Read/ReadVariableOpReadVariableOptrue_positives_2*
_output_shapes
:d*
dtype0
t
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_nametrue_negatives
m
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes
:d*
dtype0
z
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:d*"
shared_namefalse_positives_1
s
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes
:d*
dtype0
z
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:d*"
shared_namefalse_negatives_1
s
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes
:d*
dtype0
?
@Adam/sequential_2/features/clarity_embedding/embedding_weights/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*Q
shared_nameB@Adam/sequential_2/features/clarity_embedding/embedding_weights/m
?
TAdam/sequential_2/features/clarity_embedding/embedding_weights/m/Read/ReadVariableOpReadVariableOp@Adam/sequential_2/features/clarity_embedding/embedding_weights/m*
_output_shapes

:*
dtype0
?
>Adam/sequential_2/features/color_embedding/embedding_weights/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>Adam/sequential_2/features/color_embedding/embedding_weights/m
?
RAdam/sequential_2/features/color_embedding/embedding_weights/m/Read/ReadVariableOpReadVariableOp>Adam/sequential_2/features/color_embedding/embedding_weights/m*
_output_shapes

:*
dtype0
?
%Adam/sequential_2/middlerelu/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*6
shared_name'%Adam/sequential_2/middlerelu/kernel/m
?
9Adam/sequential_2/middlerelu/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/sequential_2/middlerelu/kernel/m*
_output_shapes
:	?*
dtype0
?
#Adam/sequential_2/middlerelu/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/sequential_2/middlerelu/bias/m
?
7Adam/sequential_2/middlerelu/bias/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_2/middlerelu/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/sequential_2/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/sequential_2/dense_2/kernel/m
?
6Adam/sequential_2/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/sequential_2/dense_2/kernel/m*
_output_shapes
:	?*
dtype0
?
 Adam/sequential_2/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/sequential_2/dense_2/bias/m
?
4Adam/sequential_2/dense_2/bias/m/Read/ReadVariableOpReadVariableOp Adam/sequential_2/dense_2/bias/m*
_output_shapes
:*
dtype0
?
@Adam/sequential_2/features/clarity_embedding/embedding_weights/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*Q
shared_nameB@Adam/sequential_2/features/clarity_embedding/embedding_weights/v
?
TAdam/sequential_2/features/clarity_embedding/embedding_weights/v/Read/ReadVariableOpReadVariableOp@Adam/sequential_2/features/clarity_embedding/embedding_weights/v*
_output_shapes

:*
dtype0
?
>Adam/sequential_2/features/color_embedding/embedding_weights/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*O
shared_name@>Adam/sequential_2/features/color_embedding/embedding_weights/v
?
RAdam/sequential_2/features/color_embedding/embedding_weights/v/Read/ReadVariableOpReadVariableOp>Adam/sequential_2/features/color_embedding/embedding_weights/v*
_output_shapes

:*
dtype0
?
%Adam/sequential_2/middlerelu/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*6
shared_name'%Adam/sequential_2/middlerelu/kernel/v
?
9Adam/sequential_2/middlerelu/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/sequential_2/middlerelu/kernel/v*
_output_shapes
:	?*
dtype0
?
#Adam/sequential_2/middlerelu/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/sequential_2/middlerelu/bias/v
?
7Adam/sequential_2/middlerelu/bias/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_2/middlerelu/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/sequential_2/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*3
shared_name$"Adam/sequential_2/dense_2/kernel/v
?
6Adam/sequential_2/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/sequential_2/dense_2/kernel/v*
_output_shapes
:	?*
dtype0
?
 Adam/sequential_2/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/sequential_2/dense_2/bias/v
?
4Adam/sequential_2/dense_2/bias/v/Read/ReadVariableOpReadVariableOp Adam/sequential_2/dense_2/bias/v*
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
GPU 2J 8? *$
fR
__inference_<lambda>_384616
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
GPU 2J 8? *$
fR
__inference_<lambda>_384624
B
NoOpNoOp^StatefulPartitionedCall^StatefulPartitionedCall_1
??
Const_6Const"/device:CPU:0*
_output_shapes
: *
dtype0*?>
value?>B?> B?>
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
,learning_ratemlmmmnmo mp!mqvrvsvtvu vv!vw*
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
VARIABLE_VALUE9sequential_2/features/clarity_embedding/embedding_weightsTlayer_with_weights-0/clarity_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE7sequential_2/features/color_embedding/embedding_weightsRlayer_with_weights-0/color_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUE*
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
nh
VARIABLE_VALUEsequential_2/middlerelu/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEsequential_2/middlerelu/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEsequential_2/dense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEsequential_2/dense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
'
D0
E1
F2
G3
H4*
* 
* 
* 

Iclarity_lookup* 

Jcolor_lookup* 
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
	Ktotal
	Lcount
M	variables
N	keras_api*
H
	Ototal
	Pcount
Q
_fn_kwargs
R	variables
S	keras_api*
[
T
thresholds
Utrue_positives
Vfalse_positives
W	variables
X	keras_api*
[
Y
thresholds
Ztrue_positives
[false_negatives
\	variables
]	keras_api*
t
^true_positives
_true_negatives
`false_positives
afalse_negatives
b	variables
c	keras_api*
R
d_initializer
e_create_resource
f_initialize
g_destroy_resource* 
R
h_initializer
i_create_resource
j_initialize
k_destroy_resource* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

K0
L1*

M	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

O0
P1*

R	variables*
* 
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

U0
V1*

W	variables*
* 
ga
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*

Z0
[1*

\	variables*
ga
VARIABLE_VALUEtrue_positives_2=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/4/true_negatives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
 
^0
_1
`2
a3*

b	variables*
* 
* 
* 
* 
* 
* 
* 
* 
??
VARIABLE_VALUE@Adam/sequential_2/features/clarity_embedding/embedding_weights/mplayer_with_weights-0/clarity_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE>Adam/sequential_2/features/color_embedding/embedding_weights/mnlayer_with_weights-0/color_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE%Adam/sequential_2/middlerelu/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/sequential_2/middlerelu/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/sequential_2/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/sequential_2/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE@Adam/sequential_2/features/clarity_embedding/embedding_weights/vplayer_with_weights-0/clarity_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE>Adam/sequential_2/features/color_embedding/embedding_weights/vnlayer_with_weights-0/color_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE%Adam/sequential_2/middlerelu/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/sequential_2/middlerelu/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/sequential_2/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/sequential_2/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
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
hash_tableConst9sequential_2/features/clarity_embedding/embedding_weightshash_table_1Const_17sequential_2/features/color_embedding/embedding_weightssequential_2/middlerelu/kernelsequential_2/middlerelu/biassequential_2/dense_2/kernelsequential_2/dense_2/bias*
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
GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_384110
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameMsequential_2/features/clarity_embedding/embedding_weights/Read/ReadVariableOpKsequential_2/features/color_embedding/embedding_weights/Read/ReadVariableOp2sequential_2/middlerelu/kernel/Read/ReadVariableOp0sequential_2/middlerelu/bias/Read/ReadVariableOp/sequential_2/dense_2/kernel/Read/ReadVariableOp-sequential_2/dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_2/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOpTAdam/sequential_2/features/clarity_embedding/embedding_weights/m/Read/ReadVariableOpRAdam/sequential_2/features/color_embedding/embedding_weights/m/Read/ReadVariableOp9Adam/sequential_2/middlerelu/kernel/m/Read/ReadVariableOp7Adam/sequential_2/middlerelu/bias/m/Read/ReadVariableOp6Adam/sequential_2/dense_2/kernel/m/Read/ReadVariableOp4Adam/sequential_2/dense_2/bias/m/Read/ReadVariableOpTAdam/sequential_2/features/clarity_embedding/embedding_weights/v/Read/ReadVariableOpRAdam/sequential_2/features/color_embedding/embedding_weights/v/Read/ReadVariableOp9Adam/sequential_2/middlerelu/kernel/v/Read/ReadVariableOp7Adam/sequential_2/middlerelu/bias/v/Read/ReadVariableOp6Adam/sequential_2/dense_2/kernel/v/Read/ReadVariableOp4Adam/sequential_2/dense_2/bias/v/Read/ReadVariableOpConst_6*0
Tin)
'2%	*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_384770
?

StatefulPartitionedCall_4StatefulPartitionedCallsaver_filename9sequential_2/features/clarity_embedding/embedding_weights7sequential_2/features/color_embedding/embedding_weightssequential_2/middlerelu/kernelsequential_2/middlerelu/biassequential_2/dense_2/kernelsequential_2/dense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1true_positivesfalse_positivestrue_positives_1false_negativestrue_positives_2true_negativesfalse_positives_1false_negatives_1@Adam/sequential_2/features/clarity_embedding/embedding_weights/m>Adam/sequential_2/features/color_embedding/embedding_weights/m%Adam/sequential_2/middlerelu/kernel/m#Adam/sequential_2/middlerelu/bias/m"Adam/sequential_2/dense_2/kernel/m Adam/sequential_2/dense_2/bias/m@Adam/sequential_2/features/clarity_embedding/embedding_weights/v>Adam/sequential_2/features/color_embedding/embedding_weights/v%Adam/sequential_2/middlerelu/kernel/v#Adam/sequential_2/middlerelu/bias/v"Adam/sequential_2/dense_2/kernel/v Adam/sequential_2/dense_2/bias/v*/
Tin(
&2$*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_384885ӛ
?
?
__inference__initializer_3846035
1table_init361038_lookuptableimportv2_table_handle-
)table_init361038_lookuptableimportv2_keys	/
+table_init361038_lookuptableimportv2_values	
identity??$table_init361038/LookupTableImportV2?
$table_init361038/LookupTableImportV2LookupTableImportV21table_init361038_lookuptableimportv2_table_handle)table_init361038_lookuptableimportv2_keys+table_init361038_lookuptableimportv2_values*	
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
: m
NoOpNoOp%^table_init361038/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2L
$table_init361038/LookupTableImportV2$table_init361038/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
H__inference_sequential_2_layer_call_and_return_conditional_losses_383119

inputs
inputs_1	
inputs_2	
inputs_3	
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
features_383071
features_383073	!
features_383075:
features_383077
features_383079	!
features_383081:$
middlerelu_383096:	? 
middlerelu_383098:	?!
dense_2_383113:	?
dense_2_383115:
identity??dense_2/StatefulPartitionedCall? features/StatefulPartitionedCall?"middlerelu/StatefulPartitionedCall?
 features/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8features_383071features_383073features_383075features_383077features_383079features_383081*
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
GPU 2J 8? *M
fHRF
D__inference_features_layer_call_and_return_conditional_losses_383070?
"middlerelu/StatefulPartitionedCallStatefulPartitionedCall)features/StatefulPartitionedCall:output:0middlerelu_383096middlerelu_383098*
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
GPU 2J 8? *O
fJRH
F__inference_middlerelu_layer_call_and_return_conditional_losses_383095?
dense_2/StatefulPartitionedCallStatefulPartitionedCall+middlerelu/StatefulPartitionedCall:output:0dense_2_383113dense_2_383115*
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
GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_383112w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall!^features/StatefulPartitionedCall#^middlerelu/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2D
 features/StatefulPartitionedCall features/StatefulPartitionedCall2H
"middlerelu/StatefulPartitionedCall"middlerelu/StatefulPartitionedCall:K G
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
?
?
-__inference_sequential_2_layer_call_fn_383533	
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
GPU 2J 8? *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_383477o
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
?
?
)__inference_features_layer_call_fn_384160
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
GPU 2J 8? *M
fHRF
D__inference_features_layer_call_and_return_conditional_losses_383384o
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
??
?

!__inference__wrapped_model_382861	
carat
clarity		
color	
cut		
depth	
table
x
y
zV
Rsequential_2_features_clarity_embedding_none_lookup_lookuptablefindv2_table_handleW
Ssequential_2_features_clarity_embedding_none_lookup_lookuptablefindv2_default_value	Q
?sequential_2_features_clarity_embedding_readvariableop_resource:T
Psequential_2_features_color_embedding_none_lookup_lookuptablefindv2_table_handleU
Qsequential_2_features_color_embedding_none_lookup_lookuptablefindv2_default_value	O
=sequential_2_features_color_embedding_readvariableop_resource:I
6sequential_2_middlerelu_matmul_readvariableop_resource:	?F
7sequential_2_middlerelu_biasadd_readvariableop_resource:	?F
3sequential_2_dense_2_matmul_readvariableop_resource:	?B
4sequential_2_dense_2_biasadd_readvariableop_resource:
identity??+sequential_2/dense_2/BiasAdd/ReadVariableOp?*sequential_2/dense_2/MatMul/ReadVariableOp?Esequential_2/features/clarity_embedding/None_Lookup/LookupTableFindV2?6sequential_2/features/clarity_embedding/ReadVariableOp?Csequential_2/features/color_embedding/None_Lookup/LookupTableFindV2?4sequential_2/features/color_embedding/ReadVariableOp?.sequential_2/middlerelu/BiasAdd/ReadVariableOp?-sequential_2/middlerelu/MatMul/ReadVariableOpu
*sequential_2/features/carat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
&sequential_2/features/carat/ExpandDims
ExpandDimscarat3sequential_2/features/carat/ExpandDims/dim:output:0*
T0*'
_output_shapes
:??????????
!sequential_2/features/carat/ShapeShape/sequential_2/features/carat/ExpandDims:output:0*
T0*
_output_shapes
:y
/sequential_2/features/carat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_2/features/carat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_2/features/carat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_2/features/carat/strided_sliceStridedSlice*sequential_2/features/carat/Shape:output:08sequential_2/features/carat/strided_slice/stack:output:0:sequential_2/features/carat/strided_slice/stack_1:output:0:sequential_2/features/carat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+sequential_2/features/carat/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
)sequential_2/features/carat/Reshape/shapePack2sequential_2/features/carat/strided_slice:output:04sequential_2/features/carat/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
#sequential_2/features/carat/ReshapeReshape/sequential_2/features/carat/ExpandDims:output:02sequential_2/features/carat/Reshape/shape:output:0*
T0*'
_output_shapes
:??????????
6sequential_2/features/clarity_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
2sequential_2/features/clarity_embedding/ExpandDims
ExpandDimsclarity?sequential_2/features/clarity_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
Fsequential_2/features/clarity_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Dsequential_2/features/clarity_embedding/to_sparse_input/ignore_valueCastOsequential_2/features/clarity_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
@sequential_2/features/clarity_embedding/to_sparse_input/NotEqualNotEqual;sequential_2/features/clarity_embedding/ExpandDims:output:0Hsequential_2/features/clarity_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
?sequential_2/features/clarity_embedding/to_sparse_input/indicesWhereDsequential_2/features/clarity_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
>sequential_2/features/clarity_embedding/to_sparse_input/valuesGatherNd;sequential_2/features/clarity_embedding/ExpandDims:output:0Gsequential_2/features/clarity_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Csequential_2/features/clarity_embedding/to_sparse_input/dense_shapeShape;sequential_2/features/clarity_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
Esequential_2/features/clarity_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2Rsequential_2_features_clarity_embedding_none_lookup_lookuptablefindv2_table_handleGsequential_2/features/clarity_embedding/to_sparse_input/values:output:0Ssequential_2_features_clarity_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
6sequential_2/features/clarity_embedding/ReadVariableOpReadVariableOp?sequential_2_features_clarity_embedding_readvariableop_resource*
_output_shapes

:*
dtype0?
Msequential_2/features/clarity_embedding/clarity_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Lsequential_2/features/clarity_embedding/clarity_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
Gsequential_2/features/clarity_embedding/clarity_embedding_weights/SliceSliceLsequential_2/features/clarity_embedding/to_sparse_input/dense_shape:output:0Vsequential_2/features/clarity_embedding/clarity_embedding_weights/Slice/begin:output:0Usequential_2/features/clarity_embedding/clarity_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:?
Gsequential_2/features/clarity_embedding/clarity_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Fsequential_2/features/clarity_embedding/clarity_embedding_weights/ProdProdPsequential_2/features/clarity_embedding/clarity_embedding_weights/Slice:output:0Psequential_2/features/clarity_embedding/clarity_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ?
Rsequential_2/features/clarity_embedding/clarity_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Osequential_2/features/clarity_embedding/clarity_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Jsequential_2/features/clarity_embedding/clarity_embedding_weights/GatherV2GatherV2Lsequential_2/features/clarity_embedding/to_sparse_input/dense_shape:output:0[sequential_2/features/clarity_embedding/clarity_embedding_weights/GatherV2/indices:output:0Xsequential_2/features/clarity_embedding/clarity_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
Hsequential_2/features/clarity_embedding/clarity_embedding_weights/Cast/xPackOsequential_2/features/clarity_embedding/clarity_embedding_weights/Prod:output:0Ssequential_2/features/clarity_embedding/clarity_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
Osequential_2/features/clarity_embedding/clarity_embedding_weights/SparseReshapeSparseReshapeGsequential_2/features/clarity_embedding/to_sparse_input/indices:index:0Lsequential_2/features/clarity_embedding/to_sparse_input/dense_shape:output:0Qsequential_2/features/clarity_embedding/clarity_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
Xsequential_2/features/clarity_embedding/clarity_embedding_weights/SparseReshape/IdentityIdentityNsequential_2/features/clarity_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
Psequential_2/features/clarity_embedding/clarity_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Nsequential_2/features/clarity_embedding/clarity_embedding_weights/GreaterEqualGreaterEqualasequential_2/features/clarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0Ysequential_2/features/clarity_embedding/clarity_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
Gsequential_2/features/clarity_embedding/clarity_embedding_weights/WhereWhereRsequential_2/features/clarity_embedding/clarity_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
Osequential_2/features/clarity_embedding/clarity_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Isequential_2/features/clarity_embedding/clarity_embedding_weights/ReshapeReshapeOsequential_2/features/clarity_embedding/clarity_embedding_weights/Where:index:0Xsequential_2/features/clarity_embedding/clarity_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:??????????
Qsequential_2/features/clarity_embedding/clarity_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Lsequential_2/features/clarity_embedding/clarity_embedding_weights/GatherV2_1GatherV2`sequential_2/features/clarity_embedding/clarity_embedding_weights/SparseReshape:output_indices:0Rsequential_2/features/clarity_embedding/clarity_embedding_weights/Reshape:output:0Zsequential_2/features/clarity_embedding/clarity_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:??????????
Qsequential_2/features/clarity_embedding/clarity_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Lsequential_2/features/clarity_embedding/clarity_embedding_weights/GatherV2_2GatherV2asequential_2/features/clarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0Rsequential_2/features/clarity_embedding/clarity_embedding_weights/Reshape:output:0Zsequential_2/features/clarity_embedding/clarity_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Jsequential_2/features/clarity_embedding/clarity_embedding_weights/IdentityIdentity^sequential_2/features/clarity_embedding/clarity_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
[sequential_2/features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
isequential_2/features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsUsequential_2/features/clarity_embedding/clarity_embedding_weights/GatherV2_1:output:0Usequential_2/features/clarity_embedding/clarity_embedding_weights/GatherV2_2:output:0Ssequential_2/features/clarity_embedding/clarity_embedding_weights/Identity:output:0dsequential_2/features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
msequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
osequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
osequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
gsequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicezsequential_2/features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0vsequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0xsequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0xsequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
`sequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/UniqueUniqueysequential_2/features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
osequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*I
_class?
=;loc:@sequential_2/features/clarity_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
jsequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2>sequential_2/features/clarity_embedding/ReadVariableOp:value:0dsequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:y:0xsequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*I
_class?
=;loc:@sequential_2/features/clarity_embedding/ReadVariableOp*'
_output_shapes
:??????????
ssequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityssequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
Ysequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparseSparseSegmentMean|sequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0fsequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:idx:0psequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
Qsequential_2/features/clarity_embedding/clarity_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Ksequential_2/features/clarity_embedding/clarity_embedding_weights/Reshape_1Reshapesequential_2/features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Zsequential_2/features/clarity_embedding/clarity_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
Gsequential_2/features/clarity_embedding/clarity_embedding_weights/ShapeShapebsequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
Usequential_2/features/clarity_embedding/clarity_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Wsequential_2/features/clarity_embedding/clarity_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Wsequential_2/features/clarity_embedding/clarity_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Osequential_2/features/clarity_embedding/clarity_embedding_weights/strided_sliceStridedSlicePsequential_2/features/clarity_embedding/clarity_embedding_weights/Shape:output:0^sequential_2/features/clarity_embedding/clarity_embedding_weights/strided_slice/stack:output:0`sequential_2/features/clarity_embedding/clarity_embedding_weights/strided_slice/stack_1:output:0`sequential_2/features/clarity_embedding/clarity_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Isequential_2/features/clarity_embedding/clarity_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
Gsequential_2/features/clarity_embedding/clarity_embedding_weights/stackPackRsequential_2/features/clarity_embedding/clarity_embedding_weights/stack/0:output:0Xsequential_2/features/clarity_embedding/clarity_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
Fsequential_2/features/clarity_embedding/clarity_embedding_weights/TileTileTsequential_2/features/clarity_embedding/clarity_embedding_weights/Reshape_1:output:0Psequential_2/features/clarity_embedding/clarity_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
Lsequential_2/features/clarity_embedding/clarity_embedding_weights/zeros_like	ZerosLikebsequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
Asequential_2/features/clarity_embedding/clarity_embedding_weightsSelectOsequential_2/features/clarity_embedding/clarity_embedding_weights/Tile:output:0Psequential_2/features/clarity_embedding/clarity_embedding_weights/zeros_like:y:0bsequential_2/features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
Hsequential_2/features/clarity_embedding/clarity_embedding_weights/Cast_1CastLsequential_2/features/clarity_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Osequential_2/features/clarity_embedding/clarity_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Nsequential_2/features/clarity_embedding/clarity_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
Isequential_2/features/clarity_embedding/clarity_embedding_weights/Slice_1SliceLsequential_2/features/clarity_embedding/clarity_embedding_weights/Cast_1:y:0Xsequential_2/features/clarity_embedding/clarity_embedding_weights/Slice_1/begin:output:0Wsequential_2/features/clarity_embedding/clarity_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
Isequential_2/features/clarity_embedding/clarity_embedding_weights/Shape_1ShapeJsequential_2/features/clarity_embedding/clarity_embedding_weights:output:0*
T0*
_output_shapes
:?
Osequential_2/features/clarity_embedding/clarity_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
Nsequential_2/features/clarity_embedding/clarity_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Isequential_2/features/clarity_embedding/clarity_embedding_weights/Slice_2SliceRsequential_2/features/clarity_embedding/clarity_embedding_weights/Shape_1:output:0Xsequential_2/features/clarity_embedding/clarity_embedding_weights/Slice_2/begin:output:0Wsequential_2/features/clarity_embedding/clarity_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:?
Msequential_2/features/clarity_embedding/clarity_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Hsequential_2/features/clarity_embedding/clarity_embedding_weights/concatConcatV2Rsequential_2/features/clarity_embedding/clarity_embedding_weights/Slice_1:output:0Rsequential_2/features/clarity_embedding/clarity_embedding_weights/Slice_2:output:0Vsequential_2/features/clarity_embedding/clarity_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Ksequential_2/features/clarity_embedding/clarity_embedding_weights/Reshape_2ReshapeJsequential_2/features/clarity_embedding/clarity_embedding_weights:output:0Qsequential_2/features/clarity_embedding/clarity_embedding_weights/concat:output:0*
T0*'
_output_shapes
:??????????
-sequential_2/features/clarity_embedding/ShapeShapeTsequential_2/features/clarity_embedding/clarity_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:?
;sequential_2/features/clarity_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
=sequential_2/features/clarity_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=sequential_2/features/clarity_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5sequential_2/features/clarity_embedding/strided_sliceStridedSlice6sequential_2/features/clarity_embedding/Shape:output:0Dsequential_2/features/clarity_embedding/strided_slice/stack:output:0Fsequential_2/features/clarity_embedding/strided_slice/stack_1:output:0Fsequential_2/features/clarity_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
7sequential_2/features/clarity_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
5sequential_2/features/clarity_embedding/Reshape/shapePack>sequential_2/features/clarity_embedding/strided_slice:output:0@sequential_2/features/clarity_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
/sequential_2/features/clarity_embedding/ReshapeReshapeTsequential_2/features/clarity_embedding/clarity_embedding_weights/Reshape_2:output:0>sequential_2/features/clarity_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????
4sequential_2/features/color_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
0sequential_2/features/color_embedding/ExpandDims
ExpandDimscolor=sequential_2/features/color_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
Dsequential_2/features/color_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Bsequential_2/features/color_embedding/to_sparse_input/ignore_valueCastMsequential_2/features/color_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
>sequential_2/features/color_embedding/to_sparse_input/NotEqualNotEqual9sequential_2/features/color_embedding/ExpandDims:output:0Fsequential_2/features/color_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
=sequential_2/features/color_embedding/to_sparse_input/indicesWhereBsequential_2/features/color_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
<sequential_2/features/color_embedding/to_sparse_input/valuesGatherNd9sequential_2/features/color_embedding/ExpandDims:output:0Esequential_2/features/color_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Asequential_2/features/color_embedding/to_sparse_input/dense_shapeShape9sequential_2/features/color_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
Csequential_2/features/color_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2Psequential_2_features_color_embedding_none_lookup_lookuptablefindv2_table_handleEsequential_2/features/color_embedding/to_sparse_input/values:output:0Qsequential_2_features_color_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
4sequential_2/features/color_embedding/ReadVariableOpReadVariableOp=sequential_2_features_color_embedding_readvariableop_resource*
_output_shapes

:*
dtype0?
Isequential_2/features/color_embedding/color_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Hsequential_2/features/color_embedding/color_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
Csequential_2/features/color_embedding/color_embedding_weights/SliceSliceJsequential_2/features/color_embedding/to_sparse_input/dense_shape:output:0Rsequential_2/features/color_embedding/color_embedding_weights/Slice/begin:output:0Qsequential_2/features/color_embedding/color_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:?
Csequential_2/features/color_embedding/color_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
Bsequential_2/features/color_embedding/color_embedding_weights/ProdProdLsequential_2/features/color_embedding/color_embedding_weights/Slice:output:0Lsequential_2/features/color_embedding/color_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ?
Nsequential_2/features/color_embedding/color_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Ksequential_2/features/color_embedding/color_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Fsequential_2/features/color_embedding/color_embedding_weights/GatherV2GatherV2Jsequential_2/features/color_embedding/to_sparse_input/dense_shape:output:0Wsequential_2/features/color_embedding/color_embedding_weights/GatherV2/indices:output:0Tsequential_2/features/color_embedding/color_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
Dsequential_2/features/color_embedding/color_embedding_weights/Cast/xPackKsequential_2/features/color_embedding/color_embedding_weights/Prod:output:0Osequential_2/features/color_embedding/color_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
Ksequential_2/features/color_embedding/color_embedding_weights/SparseReshapeSparseReshapeEsequential_2/features/color_embedding/to_sparse_input/indices:index:0Jsequential_2/features/color_embedding/to_sparse_input/dense_shape:output:0Msequential_2/features/color_embedding/color_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
Tsequential_2/features/color_embedding/color_embedding_weights/SparseReshape/IdentityIdentityLsequential_2/features/color_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
Lsequential_2/features/color_embedding/color_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Jsequential_2/features/color_embedding/color_embedding_weights/GreaterEqualGreaterEqual]sequential_2/features/color_embedding/color_embedding_weights/SparseReshape/Identity:output:0Usequential_2/features/color_embedding/color_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
Csequential_2/features/color_embedding/color_embedding_weights/WhereWhereNsequential_2/features/color_embedding/color_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
Ksequential_2/features/color_embedding/color_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Esequential_2/features/color_embedding/color_embedding_weights/ReshapeReshapeKsequential_2/features/color_embedding/color_embedding_weights/Where:index:0Tsequential_2/features/color_embedding/color_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:??????????
Msequential_2/features/color_embedding/color_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Hsequential_2/features/color_embedding/color_embedding_weights/GatherV2_1GatherV2\sequential_2/features/color_embedding/color_embedding_weights/SparseReshape:output_indices:0Nsequential_2/features/color_embedding/color_embedding_weights/Reshape:output:0Vsequential_2/features/color_embedding/color_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:??????????
Msequential_2/features/color_embedding/color_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Hsequential_2/features/color_embedding/color_embedding_weights/GatherV2_2GatherV2]sequential_2/features/color_embedding/color_embedding_weights/SparseReshape/Identity:output:0Nsequential_2/features/color_embedding/color_embedding_weights/Reshape:output:0Vsequential_2/features/color_embedding/color_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
Fsequential_2/features/color_embedding/color_embedding_weights/IdentityIdentityZsequential_2/features/color_embedding/color_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
Wsequential_2/features/color_embedding/color_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
esequential_2/features/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsQsequential_2/features/color_embedding/color_embedding_weights/GatherV2_1:output:0Qsequential_2/features/color_embedding/color_embedding_weights/GatherV2_2:output:0Osequential_2/features/color_embedding/color_embedding_weights/Identity:output:0`sequential_2/features/color_embedding/color_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
isequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
ksequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
ksequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
csequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicevsequential_2/features/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0rsequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0tsequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0tsequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
\sequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse/UniqueUniqueusequential_2/features/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
ksequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*G
_class=
;9loc:@sequential_2/features/color_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
fsequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2<sequential_2/features/color_embedding/ReadVariableOp:value:0`sequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:y:0tsequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*G
_class=
;9loc:@sequential_2/features/color_embedding/ReadVariableOp*'
_output_shapes
:??????????
osequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityosequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
Usequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparseSparseSegmentMeanxsequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0bsequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:idx:0lsequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
Msequential_2/features/color_embedding/color_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Gsequential_2/features/color_embedding/color_embedding_weights/Reshape_1Reshape{sequential_2/features/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Vsequential_2/features/color_embedding/color_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
Csequential_2/features/color_embedding/color_embedding_weights/ShapeShape^sequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
Qsequential_2/features/color_embedding/color_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Ssequential_2/features/color_embedding/color_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Ssequential_2/features/color_embedding/color_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Ksequential_2/features/color_embedding/color_embedding_weights/strided_sliceStridedSliceLsequential_2/features/color_embedding/color_embedding_weights/Shape:output:0Zsequential_2/features/color_embedding/color_embedding_weights/strided_slice/stack:output:0\sequential_2/features/color_embedding/color_embedding_weights/strided_slice/stack_1:output:0\sequential_2/features/color_embedding/color_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
Esequential_2/features/color_embedding/color_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
Csequential_2/features/color_embedding/color_embedding_weights/stackPackNsequential_2/features/color_embedding/color_embedding_weights/stack/0:output:0Tsequential_2/features/color_embedding/color_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
Bsequential_2/features/color_embedding/color_embedding_weights/TileTilePsequential_2/features/color_embedding/color_embedding_weights/Reshape_1:output:0Lsequential_2/features/color_embedding/color_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
Hsequential_2/features/color_embedding/color_embedding_weights/zeros_like	ZerosLike^sequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
=sequential_2/features/color_embedding/color_embedding_weightsSelectKsequential_2/features/color_embedding/color_embedding_weights/Tile:output:0Lsequential_2/features/color_embedding/color_embedding_weights/zeros_like:y:0^sequential_2/features/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
Dsequential_2/features/color_embedding/color_embedding_weights/Cast_1CastJsequential_2/features/color_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Ksequential_2/features/color_embedding/color_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Jsequential_2/features/color_embedding/color_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
Esequential_2/features/color_embedding/color_embedding_weights/Slice_1SliceHsequential_2/features/color_embedding/color_embedding_weights/Cast_1:y:0Tsequential_2/features/color_embedding/color_embedding_weights/Slice_1/begin:output:0Ssequential_2/features/color_embedding/color_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
Esequential_2/features/color_embedding/color_embedding_weights/Shape_1ShapeFsequential_2/features/color_embedding/color_embedding_weights:output:0*
T0*
_output_shapes
:?
Ksequential_2/features/color_embedding/color_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
Jsequential_2/features/color_embedding/color_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
Esequential_2/features/color_embedding/color_embedding_weights/Slice_2SliceNsequential_2/features/color_embedding/color_embedding_weights/Shape_1:output:0Tsequential_2/features/color_embedding/color_embedding_weights/Slice_2/begin:output:0Ssequential_2/features/color_embedding/color_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:?
Isequential_2/features/color_embedding/color_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Dsequential_2/features/color_embedding/color_embedding_weights/concatConcatV2Nsequential_2/features/color_embedding/color_embedding_weights/Slice_1:output:0Nsequential_2/features/color_embedding/color_embedding_weights/Slice_2:output:0Rsequential_2/features/color_embedding/color_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Gsequential_2/features/color_embedding/color_embedding_weights/Reshape_2ReshapeFsequential_2/features/color_embedding/color_embedding_weights:output:0Msequential_2/features/color_embedding/color_embedding_weights/concat:output:0*
T0*'
_output_shapes
:??????????
+sequential_2/features/color_embedding/ShapeShapePsequential_2/features/color_embedding/color_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:?
9sequential_2/features/color_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
;sequential_2/features/color_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
;sequential_2/features/color_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
3sequential_2/features/color_embedding/strided_sliceStridedSlice4sequential_2/features/color_embedding/Shape:output:0Bsequential_2/features/color_embedding/strided_slice/stack:output:0Dsequential_2/features/color_embedding/strided_slice/stack_1:output:0Dsequential_2/features/color_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5sequential_2/features/color_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
3sequential_2/features/color_embedding/Reshape/shapePack<sequential_2/features/color_embedding/strided_slice:output:0>sequential_2/features/color_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
-sequential_2/features/color_embedding/ReshapeReshapePsequential_2/features/color_embedding/color_embedding_weights/Reshape_2:output:0<sequential_2/features/color_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????l
!sequential_2/features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
sequential_2/features/concatConcatV2,sequential_2/features/carat/Reshape:output:08sequential_2/features/clarity_embedding/Reshape:output:06sequential_2/features/color_embedding/Reshape:output:0*sequential_2/features/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
-sequential_2/middlerelu/MatMul/ReadVariableOpReadVariableOp6sequential_2_middlerelu_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
sequential_2/middlerelu/MatMulMatMul%sequential_2/features/concat:output:05sequential_2/middlerelu/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
.sequential_2/middlerelu/BiasAdd/ReadVariableOpReadVariableOp7sequential_2_middlerelu_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_2/middlerelu/BiasAddBiasAdd(sequential_2/middlerelu/MatMul:product:06sequential_2/middlerelu/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential_2/middlerelu/ReluRelu(sequential_2/middlerelu/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
*sequential_2/dense_2/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
sequential_2/dense_2/MatMulMatMul*sequential_2/middlerelu/Relu:activations:02sequential_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_2/dense_2/BiasAddBiasAdd%sequential_2/dense_2/MatMul:product:03sequential_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_2/dense_2/SigmoidSigmoid%sequential_2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????o
IdentityIdentity sequential_2/dense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^sequential_2/dense_2/BiasAdd/ReadVariableOp+^sequential_2/dense_2/MatMul/ReadVariableOpF^sequential_2/features/clarity_embedding/None_Lookup/LookupTableFindV27^sequential_2/features/clarity_embedding/ReadVariableOpD^sequential_2/features/color_embedding/None_Lookup/LookupTableFindV25^sequential_2/features/color_embedding/ReadVariableOp/^sequential_2/middlerelu/BiasAdd/ReadVariableOp.^sequential_2/middlerelu/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2Z
+sequential_2/dense_2/BiasAdd/ReadVariableOp+sequential_2/dense_2/BiasAdd/ReadVariableOp2X
*sequential_2/dense_2/MatMul/ReadVariableOp*sequential_2/dense_2/MatMul/ReadVariableOp2?
Esequential_2/features/clarity_embedding/None_Lookup/LookupTableFindV2Esequential_2/features/clarity_embedding/None_Lookup/LookupTableFindV22p
6sequential_2/features/clarity_embedding/ReadVariableOp6sequential_2/features/clarity_embedding/ReadVariableOp2?
Csequential_2/features/color_embedding/None_Lookup/LookupTableFindV2Csequential_2/features/color_embedding/None_Lookup/LookupTableFindV22l
4sequential_2/features/color_embedding/ReadVariableOp4sequential_2/features/color_embedding/ReadVariableOp2`
.sequential_2/middlerelu/BiasAdd/ReadVariableOp.sequential_2/middlerelu/BiasAdd/ReadVariableOp2^
-sequential_2/middlerelu/MatMul/ReadVariableOp-sequential_2/middlerelu/MatMul/ReadVariableOp:J F
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
F__inference_middlerelu_layer_call_and_return_conditional_losses_384552

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
?	
H__inference_sequential_2_layer_call_and_return_conditional_losses_383875
inputs_carat
inputs_clarity	
inputs_color	

inputs_cut	
inputs_depth
inputs_table
inputs_x
inputs_y
inputs_zI
Efeatures_clarity_embedding_none_lookup_lookuptablefindv2_table_handleJ
Ffeatures_clarity_embedding_none_lookup_lookuptablefindv2_default_value	D
2features_clarity_embedding_readvariableop_resource:G
Cfeatures_color_embedding_none_lookup_lookuptablefindv2_table_handleH
Dfeatures_color_embedding_none_lookup_lookuptablefindv2_default_value	B
0features_color_embedding_readvariableop_resource:<
)middlerelu_matmul_readvariableop_resource:	?9
*middlerelu_biasadd_readvariableop_resource:	?9
&dense_2_matmul_readvariableop_resource:	?5
'dense_2_biasadd_readvariableop_resource:
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?8features/clarity_embedding/None_Lookup/LookupTableFindV2?)features/clarity_embedding/ReadVariableOp?6features/color_embedding/None_Lookup/LookupTableFindV2?'features/color_embedding/ReadVariableOp?!middlerelu/BiasAdd/ReadVariableOp? middlerelu/MatMul/ReadVariableOph
features/carat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
features/carat/ExpandDims
ExpandDimsinputs_carat&features/carat/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????f
features/carat/ShapeShape"features/carat/ExpandDims:output:0*
T0*
_output_shapes
:l
"features/carat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$features/carat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$features/carat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
features/carat/strided_sliceStridedSlicefeatures/carat/Shape:output:0+features/carat/strided_slice/stack:output:0-features/carat/strided_slice/stack_1:output:0-features/carat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
features/carat/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
features/carat/Reshape/shapePack%features/carat/strided_slice:output:0'features/carat/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
features/carat/ReshapeReshape"features/carat/ExpandDims:output:0%features/carat/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????t
)features/clarity_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%features/clarity_embedding/ExpandDims
ExpandDimsinputs_clarity2features/clarity_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
9features/clarity_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
7features/clarity_embedding/to_sparse_input/ignore_valueCastBfeatures/clarity_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
3features/clarity_embedding/to_sparse_input/NotEqualNotEqual.features/clarity_embedding/ExpandDims:output:0;features/clarity_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
2features/clarity_embedding/to_sparse_input/indicesWhere7features/clarity_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
1features/clarity_embedding/to_sparse_input/valuesGatherNd.features/clarity_embedding/ExpandDims:output:0:features/clarity_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
6features/clarity_embedding/to_sparse_input/dense_shapeShape.features/clarity_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
8features/clarity_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2Efeatures_clarity_embedding_none_lookup_lookuptablefindv2_table_handle:features/clarity_embedding/to_sparse_input/values:output:0Ffeatures_clarity_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
)features/clarity_embedding/ReadVariableOpReadVariableOp2features_clarity_embedding_readvariableop_resource*
_output_shapes

:*
dtype0?
@features/clarity_embedding/clarity_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
?features/clarity_embedding/clarity_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
:features/clarity_embedding/clarity_embedding_weights/SliceSlice?features/clarity_embedding/to_sparse_input/dense_shape:output:0Ifeatures/clarity_embedding/clarity_embedding_weights/Slice/begin:output:0Hfeatures/clarity_embedding/clarity_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:?
:features/clarity_embedding/clarity_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
9features/clarity_embedding/clarity_embedding_weights/ProdProdCfeatures/clarity_embedding/clarity_embedding_weights/Slice:output:0Cfeatures/clarity_embedding/clarity_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ?
Efeatures/clarity_embedding/clarity_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Bfeatures/clarity_embedding/clarity_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
=features/clarity_embedding/clarity_embedding_weights/GatherV2GatherV2?features/clarity_embedding/to_sparse_input/dense_shape:output:0Nfeatures/clarity_embedding/clarity_embedding_weights/GatherV2/indices:output:0Kfeatures/clarity_embedding/clarity_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
;features/clarity_embedding/clarity_embedding_weights/Cast/xPackBfeatures/clarity_embedding/clarity_embedding_weights/Prod:output:0Ffeatures/clarity_embedding/clarity_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
Bfeatures/clarity_embedding/clarity_embedding_weights/SparseReshapeSparseReshape:features/clarity_embedding/to_sparse_input/indices:index:0?features/clarity_embedding/to_sparse_input/dense_shape:output:0Dfeatures/clarity_embedding/clarity_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
Kfeatures/clarity_embedding/clarity_embedding_weights/SparseReshape/IdentityIdentityAfeatures/clarity_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
Cfeatures/clarity_embedding/clarity_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Afeatures/clarity_embedding/clarity_embedding_weights/GreaterEqualGreaterEqualTfeatures/clarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0Lfeatures/clarity_embedding/clarity_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
:features/clarity_embedding/clarity_embedding_weights/WhereWhereEfeatures/clarity_embedding/clarity_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
Bfeatures/clarity_embedding/clarity_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
<features/clarity_embedding/clarity_embedding_weights/ReshapeReshapeBfeatures/clarity_embedding/clarity_embedding_weights/Where:index:0Kfeatures/clarity_embedding/clarity_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:??????????
Dfeatures/clarity_embedding/clarity_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
?features/clarity_embedding/clarity_embedding_weights/GatherV2_1GatherV2Sfeatures/clarity_embedding/clarity_embedding_weights/SparseReshape:output_indices:0Efeatures/clarity_embedding/clarity_embedding_weights/Reshape:output:0Mfeatures/clarity_embedding/clarity_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:??????????
Dfeatures/clarity_embedding/clarity_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
?features/clarity_embedding/clarity_embedding_weights/GatherV2_2GatherV2Tfeatures/clarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0Efeatures/clarity_embedding/clarity_embedding_weights/Reshape:output:0Mfeatures/clarity_embedding/clarity_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
=features/clarity_embedding/clarity_embedding_weights/IdentityIdentityQfeatures/clarity_embedding/clarity_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
Nfeatures/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
\features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsHfeatures/clarity_embedding/clarity_embedding_weights/GatherV2_1:output:0Hfeatures/clarity_embedding/clarity_embedding_weights/GatherV2_2:output:0Ffeatures/clarity_embedding/clarity_embedding_weights/Identity:output:0Wfeatures/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
`features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
bfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
bfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Zfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicemfeatures/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0ifeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0kfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0kfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Sfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/UniqueUniquelfeatures/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
bfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*<
_class2
0.loc:@features/clarity_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
]features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV21features/clarity_embedding/ReadVariableOp:value:0Wfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:y:0kfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*<
_class2
0.loc:@features/clarity_embedding/ReadVariableOp*'
_output_shapes
:??????????
ffeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityffeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
Lfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparseSparseSegmentMeanofeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0Yfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:idx:0cfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
Dfeatures/clarity_embedding/clarity_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
>features/clarity_embedding/clarity_embedding_weights/Reshape_1Reshaperfeatures/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Mfeatures/clarity_embedding/clarity_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
:features/clarity_embedding/clarity_embedding_weights/ShapeShapeUfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
Hfeatures/clarity_embedding/clarity_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Jfeatures/clarity_embedding/clarity_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Jfeatures/clarity_embedding/clarity_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Bfeatures/clarity_embedding/clarity_embedding_weights/strided_sliceStridedSliceCfeatures/clarity_embedding/clarity_embedding_weights/Shape:output:0Qfeatures/clarity_embedding/clarity_embedding_weights/strided_slice/stack:output:0Sfeatures/clarity_embedding/clarity_embedding_weights/strided_slice/stack_1:output:0Sfeatures/clarity_embedding/clarity_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<features/clarity_embedding/clarity_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
:features/clarity_embedding/clarity_embedding_weights/stackPackEfeatures/clarity_embedding/clarity_embedding_weights/stack/0:output:0Kfeatures/clarity_embedding/clarity_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
9features/clarity_embedding/clarity_embedding_weights/TileTileGfeatures/clarity_embedding/clarity_embedding_weights/Reshape_1:output:0Cfeatures/clarity_embedding/clarity_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
?features/clarity_embedding/clarity_embedding_weights/zeros_like	ZerosLikeUfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
4features/clarity_embedding/clarity_embedding_weightsSelectBfeatures/clarity_embedding/clarity_embedding_weights/Tile:output:0Cfeatures/clarity_embedding/clarity_embedding_weights/zeros_like:y:0Ufeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
;features/clarity_embedding/clarity_embedding_weights/Cast_1Cast?features/clarity_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Bfeatures/clarity_embedding/clarity_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Afeatures/clarity_embedding/clarity_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
<features/clarity_embedding/clarity_embedding_weights/Slice_1Slice?features/clarity_embedding/clarity_embedding_weights/Cast_1:y:0Kfeatures/clarity_embedding/clarity_embedding_weights/Slice_1/begin:output:0Jfeatures/clarity_embedding/clarity_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
<features/clarity_embedding/clarity_embedding_weights/Shape_1Shape=features/clarity_embedding/clarity_embedding_weights:output:0*
T0*
_output_shapes
:?
Bfeatures/clarity_embedding/clarity_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
Afeatures/clarity_embedding/clarity_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
<features/clarity_embedding/clarity_embedding_weights/Slice_2SliceEfeatures/clarity_embedding/clarity_embedding_weights/Shape_1:output:0Kfeatures/clarity_embedding/clarity_embedding_weights/Slice_2/begin:output:0Jfeatures/clarity_embedding/clarity_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:?
@features/clarity_embedding/clarity_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
;features/clarity_embedding/clarity_embedding_weights/concatConcatV2Efeatures/clarity_embedding/clarity_embedding_weights/Slice_1:output:0Efeatures/clarity_embedding/clarity_embedding_weights/Slice_2:output:0Ifeatures/clarity_embedding/clarity_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
>features/clarity_embedding/clarity_embedding_weights/Reshape_2Reshape=features/clarity_embedding/clarity_embedding_weights:output:0Dfeatures/clarity_embedding/clarity_embedding_weights/concat:output:0*
T0*'
_output_shapes
:??????????
 features/clarity_embedding/ShapeShapeGfeatures/clarity_embedding/clarity_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:x
.features/clarity_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0features/clarity_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0features/clarity_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(features/clarity_embedding/strided_sliceStridedSlice)features/clarity_embedding/Shape:output:07features/clarity_embedding/strided_slice/stack:output:09features/clarity_embedding/strided_slice/stack_1:output:09features/clarity_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*features/clarity_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
(features/clarity_embedding/Reshape/shapePack1features/clarity_embedding/strided_slice:output:03features/clarity_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
"features/clarity_embedding/ReshapeReshapeGfeatures/clarity_embedding/clarity_embedding_weights/Reshape_2:output:01features/clarity_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????r
'features/color_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#features/color_embedding/ExpandDims
ExpandDimsinputs_color0features/color_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
7features/color_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
5features/color_embedding/to_sparse_input/ignore_valueCast@features/color_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
1features/color_embedding/to_sparse_input/NotEqualNotEqual,features/color_embedding/ExpandDims:output:09features/color_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
0features/color_embedding/to_sparse_input/indicesWhere5features/color_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
/features/color_embedding/to_sparse_input/valuesGatherNd,features/color_embedding/ExpandDims:output:08features/color_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
4features/color_embedding/to_sparse_input/dense_shapeShape,features/color_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
6features/color_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2Cfeatures_color_embedding_none_lookup_lookuptablefindv2_table_handle8features/color_embedding/to_sparse_input/values:output:0Dfeatures_color_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
'features/color_embedding/ReadVariableOpReadVariableOp0features_color_embedding_readvariableop_resource*
_output_shapes

:*
dtype0?
<features/color_embedding/color_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
;features/color_embedding/color_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
6features/color_embedding/color_embedding_weights/SliceSlice=features/color_embedding/to_sparse_input/dense_shape:output:0Efeatures/color_embedding/color_embedding_weights/Slice/begin:output:0Dfeatures/color_embedding/color_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:?
6features/color_embedding/color_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
5features/color_embedding/color_embedding_weights/ProdProd?features/color_embedding/color_embedding_weights/Slice:output:0?features/color_embedding/color_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ?
Afeatures/color_embedding/color_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :?
>features/color_embedding/color_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
9features/color_embedding/color_embedding_weights/GatherV2GatherV2=features/color_embedding/to_sparse_input/dense_shape:output:0Jfeatures/color_embedding/color_embedding_weights/GatherV2/indices:output:0Gfeatures/color_embedding/color_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
7features/color_embedding/color_embedding_weights/Cast/xPack>features/color_embedding/color_embedding_weights/Prod:output:0Bfeatures/color_embedding/color_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
>features/color_embedding/color_embedding_weights/SparseReshapeSparseReshape8features/color_embedding/to_sparse_input/indices:index:0=features/color_embedding/to_sparse_input/dense_shape:output:0@features/color_embedding/color_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
Gfeatures/color_embedding/color_embedding_weights/SparseReshape/IdentityIdentity?features/color_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
?features/color_embedding/color_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
=features/color_embedding/color_embedding_weights/GreaterEqualGreaterEqualPfeatures/color_embedding/color_embedding_weights/SparseReshape/Identity:output:0Hfeatures/color_embedding/color_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
6features/color_embedding/color_embedding_weights/WhereWhereAfeatures/color_embedding/color_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
>features/color_embedding/color_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
8features/color_embedding/color_embedding_weights/ReshapeReshape>features/color_embedding/color_embedding_weights/Where:index:0Gfeatures/color_embedding/color_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:??????????
@features/color_embedding/color_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
;features/color_embedding/color_embedding_weights/GatherV2_1GatherV2Ofeatures/color_embedding/color_embedding_weights/SparseReshape:output_indices:0Afeatures/color_embedding/color_embedding_weights/Reshape:output:0Ifeatures/color_embedding/color_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:??????????
@features/color_embedding/color_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
;features/color_embedding/color_embedding_weights/GatherV2_2GatherV2Pfeatures/color_embedding/color_embedding_weights/SparseReshape/Identity:output:0Afeatures/color_embedding/color_embedding_weights/Reshape:output:0Ifeatures/color_embedding/color_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
9features/color_embedding/color_embedding_weights/IdentityIdentityMfeatures/color_embedding/color_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
Jfeatures/color_embedding/color_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Xfeatures/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsDfeatures/color_embedding/color_embedding_weights/GatherV2_1:output:0Dfeatures/color_embedding/color_embedding_weights/GatherV2_2:output:0Bfeatures/color_embedding/color_embedding_weights/Identity:output:0Sfeatures/color_embedding/color_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
\features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
^features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
^features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Vfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceifeatures/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0efeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0gfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0gfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Ofeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/UniqueUniquehfeatures/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
^features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*:
_class0
.,loc:@features/color_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
Yfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2/features/color_embedding/ReadVariableOp:value:0Sfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:y:0gfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*:
_class0
.,loc:@features/color_embedding/ReadVariableOp*'
_output_shapes
:??????????
bfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitybfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
Hfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparseSparseSegmentMeankfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0Ufeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:idx:0_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
@features/color_embedding/color_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
:features/color_embedding/color_embedding_weights/Reshape_1Reshapenfeatures/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Ifeatures/color_embedding/color_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
6features/color_embedding/color_embedding_weights/ShapeShapeQfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
Dfeatures/color_embedding/color_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Ffeatures/color_embedding/color_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Ffeatures/color_embedding/color_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
>features/color_embedding/color_embedding_weights/strided_sliceStridedSlice?features/color_embedding/color_embedding_weights/Shape:output:0Mfeatures/color_embedding/color_embedding_weights/strided_slice/stack:output:0Ofeatures/color_embedding/color_embedding_weights/strided_slice/stack_1:output:0Ofeatures/color_embedding/color_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
8features/color_embedding/color_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
6features/color_embedding/color_embedding_weights/stackPackAfeatures/color_embedding/color_embedding_weights/stack/0:output:0Gfeatures/color_embedding/color_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
5features/color_embedding/color_embedding_weights/TileTileCfeatures/color_embedding/color_embedding_weights/Reshape_1:output:0?features/color_embedding/color_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
;features/color_embedding/color_embedding_weights/zeros_like	ZerosLikeQfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
0features/color_embedding/color_embedding_weightsSelect>features/color_embedding/color_embedding_weights/Tile:output:0?features/color_embedding/color_embedding_weights/zeros_like:y:0Qfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
7features/color_embedding/color_embedding_weights/Cast_1Cast=features/color_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
>features/color_embedding/color_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
=features/color_embedding/color_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
8features/color_embedding/color_embedding_weights/Slice_1Slice;features/color_embedding/color_embedding_weights/Cast_1:y:0Gfeatures/color_embedding/color_embedding_weights/Slice_1/begin:output:0Ffeatures/color_embedding/color_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
8features/color_embedding/color_embedding_weights/Shape_1Shape9features/color_embedding/color_embedding_weights:output:0*
T0*
_output_shapes
:?
>features/color_embedding/color_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
=features/color_embedding/color_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
8features/color_embedding/color_embedding_weights/Slice_2SliceAfeatures/color_embedding/color_embedding_weights/Shape_1:output:0Gfeatures/color_embedding/color_embedding_weights/Slice_2/begin:output:0Ffeatures/color_embedding/color_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:~
<features/color_embedding/color_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
7features/color_embedding/color_embedding_weights/concatConcatV2Afeatures/color_embedding/color_embedding_weights/Slice_1:output:0Afeatures/color_embedding/color_embedding_weights/Slice_2:output:0Efeatures/color_embedding/color_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
:features/color_embedding/color_embedding_weights/Reshape_2Reshape9features/color_embedding/color_embedding_weights:output:0@features/color_embedding/color_embedding_weights/concat:output:0*
T0*'
_output_shapes
:??????????
features/color_embedding/ShapeShapeCfeatures/color_embedding/color_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:v
,features/color_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.features/color_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.features/color_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&features/color_embedding/strided_sliceStridedSlice'features/color_embedding/Shape:output:05features/color_embedding/strided_slice/stack:output:07features/color_embedding/strided_slice/stack_1:output:07features/color_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(features/color_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&features/color_embedding/Reshape/shapePack/features/color_embedding/strided_slice:output:01features/color_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 features/color_embedding/ReshapeReshapeCfeatures/color_embedding/color_embedding_weights/Reshape_2:output:0/features/color_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????_
features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
features/concatConcatV2features/carat/Reshape:output:0+features/clarity_embedding/Reshape:output:0)features/color_embedding/Reshape:output:0features/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
 middlerelu/MatMul/ReadVariableOpReadVariableOp)middlerelu_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
middlerelu/MatMulMatMulfeatures/concat:output:0(middlerelu/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!middlerelu/BiasAdd/ReadVariableOpReadVariableOp*middlerelu_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
middlerelu/BiasAddBiasAddmiddlerelu/MatMul:product:0)middlerelu/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????g
middlerelu/ReluRelumiddlerelu/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_2/MatMulMatMulmiddlerelu/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp9^features/clarity_embedding/None_Lookup/LookupTableFindV2*^features/clarity_embedding/ReadVariableOp7^features/color_embedding/None_Lookup/LookupTableFindV2(^features/color_embedding/ReadVariableOp"^middlerelu/BiasAdd/ReadVariableOp!^middlerelu/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2t
8features/clarity_embedding/None_Lookup/LookupTableFindV28features/clarity_embedding/None_Lookup/LookupTableFindV22V
)features/clarity_embedding/ReadVariableOp)features/clarity_embedding/ReadVariableOp2p
6features/color_embedding/None_Lookup/LookupTableFindV26features/color_embedding/None_Lookup/LookupTableFindV22R
'features/color_embedding/ReadVariableOp'features/color_embedding/ReadVariableOp2F
!middlerelu/BiasAdd/ReadVariableOp!middlerelu/BiasAdd/ReadVariableOp2D
 middlerelu/MatMul/ReadVariableOp middlerelu/MatMul/ReadVariableOp:Q M
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
D__inference_features_layer_call_and_return_conditional_losses_383384
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
?
?
(__inference_dense_2_layer_call_fn_384561

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
GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_383112o
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
??
?	
H__inference_sequential_2_layer_call_and_return_conditional_losses_384075
inputs_carat
inputs_clarity	
inputs_color	

inputs_cut	
inputs_depth
inputs_table
inputs_x
inputs_y
inputs_zI
Efeatures_clarity_embedding_none_lookup_lookuptablefindv2_table_handleJ
Ffeatures_clarity_embedding_none_lookup_lookuptablefindv2_default_value	D
2features_clarity_embedding_readvariableop_resource:G
Cfeatures_color_embedding_none_lookup_lookuptablefindv2_table_handleH
Dfeatures_color_embedding_none_lookup_lookuptablefindv2_default_value	B
0features_color_embedding_readvariableop_resource:<
)middlerelu_matmul_readvariableop_resource:	?9
*middlerelu_biasadd_readvariableop_resource:	?9
&dense_2_matmul_readvariableop_resource:	?5
'dense_2_biasadd_readvariableop_resource:
identity??dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?8features/clarity_embedding/None_Lookup/LookupTableFindV2?)features/clarity_embedding/ReadVariableOp?6features/color_embedding/None_Lookup/LookupTableFindV2?'features/color_embedding/ReadVariableOp?!middlerelu/BiasAdd/ReadVariableOp? middlerelu/MatMul/ReadVariableOph
features/carat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
features/carat/ExpandDims
ExpandDimsinputs_carat&features/carat/ExpandDims/dim:output:0*
T0*'
_output_shapes
:?????????f
features/carat/ShapeShape"features/carat/ExpandDims:output:0*
T0*
_output_shapes
:l
"features/carat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$features/carat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$features/carat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
features/carat/strided_sliceStridedSlicefeatures/carat/Shape:output:0+features/carat/strided_slice/stack:output:0-features/carat/strided_slice/stack_1:output:0-features/carat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
features/carat/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
features/carat/Reshape/shapePack%features/carat/strided_slice:output:0'features/carat/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
features/carat/ReshapeReshape"features/carat/ExpandDims:output:0%features/carat/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????t
)features/clarity_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
%features/clarity_embedding/ExpandDims
ExpandDimsinputs_clarity2features/clarity_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
9features/clarity_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
7features/clarity_embedding/to_sparse_input/ignore_valueCastBfeatures/clarity_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
3features/clarity_embedding/to_sparse_input/NotEqualNotEqual.features/clarity_embedding/ExpandDims:output:0;features/clarity_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
2features/clarity_embedding/to_sparse_input/indicesWhere7features/clarity_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
1features/clarity_embedding/to_sparse_input/valuesGatherNd.features/clarity_embedding/ExpandDims:output:0:features/clarity_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
6features/clarity_embedding/to_sparse_input/dense_shapeShape.features/clarity_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
8features/clarity_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2Efeatures_clarity_embedding_none_lookup_lookuptablefindv2_table_handle:features/clarity_embedding/to_sparse_input/values:output:0Ffeatures_clarity_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
)features/clarity_embedding/ReadVariableOpReadVariableOp2features_clarity_embedding_readvariableop_resource*
_output_shapes

:*
dtype0?
@features/clarity_embedding/clarity_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
?features/clarity_embedding/clarity_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
:features/clarity_embedding/clarity_embedding_weights/SliceSlice?features/clarity_embedding/to_sparse_input/dense_shape:output:0Ifeatures/clarity_embedding/clarity_embedding_weights/Slice/begin:output:0Hfeatures/clarity_embedding/clarity_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:?
:features/clarity_embedding/clarity_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
9features/clarity_embedding/clarity_embedding_weights/ProdProdCfeatures/clarity_embedding/clarity_embedding_weights/Slice:output:0Cfeatures/clarity_embedding/clarity_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ?
Efeatures/clarity_embedding/clarity_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :?
Bfeatures/clarity_embedding/clarity_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
=features/clarity_embedding/clarity_embedding_weights/GatherV2GatherV2?features/clarity_embedding/to_sparse_input/dense_shape:output:0Nfeatures/clarity_embedding/clarity_embedding_weights/GatherV2/indices:output:0Kfeatures/clarity_embedding/clarity_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
;features/clarity_embedding/clarity_embedding_weights/Cast/xPackBfeatures/clarity_embedding/clarity_embedding_weights/Prod:output:0Ffeatures/clarity_embedding/clarity_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
Bfeatures/clarity_embedding/clarity_embedding_weights/SparseReshapeSparseReshape:features/clarity_embedding/to_sparse_input/indices:index:0?features/clarity_embedding/to_sparse_input/dense_shape:output:0Dfeatures/clarity_embedding/clarity_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
Kfeatures/clarity_embedding/clarity_embedding_weights/SparseReshape/IdentityIdentityAfeatures/clarity_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
Cfeatures/clarity_embedding/clarity_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Afeatures/clarity_embedding/clarity_embedding_weights/GreaterEqualGreaterEqualTfeatures/clarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0Lfeatures/clarity_embedding/clarity_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
:features/clarity_embedding/clarity_embedding_weights/WhereWhereEfeatures/clarity_embedding/clarity_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
Bfeatures/clarity_embedding/clarity_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
<features/clarity_embedding/clarity_embedding_weights/ReshapeReshapeBfeatures/clarity_embedding/clarity_embedding_weights/Where:index:0Kfeatures/clarity_embedding/clarity_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:??????????
Dfeatures/clarity_embedding/clarity_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
?features/clarity_embedding/clarity_embedding_weights/GatherV2_1GatherV2Sfeatures/clarity_embedding/clarity_embedding_weights/SparseReshape:output_indices:0Efeatures/clarity_embedding/clarity_embedding_weights/Reshape:output:0Mfeatures/clarity_embedding/clarity_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:??????????
Dfeatures/clarity_embedding/clarity_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
?features/clarity_embedding/clarity_embedding_weights/GatherV2_2GatherV2Tfeatures/clarity_embedding/clarity_embedding_weights/SparseReshape/Identity:output:0Efeatures/clarity_embedding/clarity_embedding_weights/Reshape:output:0Mfeatures/clarity_embedding/clarity_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
=features/clarity_embedding/clarity_embedding_weights/IdentityIdentityQfeatures/clarity_embedding/clarity_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
Nfeatures/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
\features/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsHfeatures/clarity_embedding/clarity_embedding_weights/GatherV2_1:output:0Hfeatures/clarity_embedding/clarity_embedding_weights/GatherV2_2:output:0Ffeatures/clarity_embedding/clarity_embedding_weights/Identity:output:0Wfeatures/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
`features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
bfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
bfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Zfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlicemfeatures/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0ifeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0kfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0kfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Sfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/UniqueUniquelfeatures/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
bfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*<
_class2
0.loc:@features/clarity_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
]features/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV21features/clarity_embedding/ReadVariableOp:value:0Wfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:y:0kfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*<
_class2
0.loc:@features/clarity_embedding/ReadVariableOp*'
_output_shapes
:??????????
ffeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentityffeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
Lfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparseSparseSegmentMeanofeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0Yfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/Unique:idx:0cfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
Dfeatures/clarity_embedding/clarity_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
>features/clarity_embedding/clarity_embedding_weights/Reshape_1Reshaperfeatures/clarity_embedding/clarity_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Mfeatures/clarity_embedding/clarity_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
:features/clarity_embedding/clarity_embedding_weights/ShapeShapeUfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
Hfeatures/clarity_embedding/clarity_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Jfeatures/clarity_embedding/clarity_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Jfeatures/clarity_embedding/clarity_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Bfeatures/clarity_embedding/clarity_embedding_weights/strided_sliceStridedSliceCfeatures/clarity_embedding/clarity_embedding_weights/Shape:output:0Qfeatures/clarity_embedding/clarity_embedding_weights/strided_slice/stack:output:0Sfeatures/clarity_embedding/clarity_embedding_weights/strided_slice/stack_1:output:0Sfeatures/clarity_embedding/clarity_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<features/clarity_embedding/clarity_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
:features/clarity_embedding/clarity_embedding_weights/stackPackEfeatures/clarity_embedding/clarity_embedding_weights/stack/0:output:0Kfeatures/clarity_embedding/clarity_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
9features/clarity_embedding/clarity_embedding_weights/TileTileGfeatures/clarity_embedding/clarity_embedding_weights/Reshape_1:output:0Cfeatures/clarity_embedding/clarity_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
?features/clarity_embedding/clarity_embedding_weights/zeros_like	ZerosLikeUfeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
4features/clarity_embedding/clarity_embedding_weightsSelectBfeatures/clarity_embedding/clarity_embedding_weights/Tile:output:0Cfeatures/clarity_embedding/clarity_embedding_weights/zeros_like:y:0Ufeatures/clarity_embedding/clarity_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
;features/clarity_embedding/clarity_embedding_weights/Cast_1Cast?features/clarity_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
Bfeatures/clarity_embedding/clarity_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
Afeatures/clarity_embedding/clarity_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
<features/clarity_embedding/clarity_embedding_weights/Slice_1Slice?features/clarity_embedding/clarity_embedding_weights/Cast_1:y:0Kfeatures/clarity_embedding/clarity_embedding_weights/Slice_1/begin:output:0Jfeatures/clarity_embedding/clarity_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
<features/clarity_embedding/clarity_embedding_weights/Shape_1Shape=features/clarity_embedding/clarity_embedding_weights:output:0*
T0*
_output_shapes
:?
Bfeatures/clarity_embedding/clarity_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
Afeatures/clarity_embedding/clarity_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
<features/clarity_embedding/clarity_embedding_weights/Slice_2SliceEfeatures/clarity_embedding/clarity_embedding_weights/Shape_1:output:0Kfeatures/clarity_embedding/clarity_embedding_weights/Slice_2/begin:output:0Jfeatures/clarity_embedding/clarity_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:?
@features/clarity_embedding/clarity_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
;features/clarity_embedding/clarity_embedding_weights/concatConcatV2Efeatures/clarity_embedding/clarity_embedding_weights/Slice_1:output:0Efeatures/clarity_embedding/clarity_embedding_weights/Slice_2:output:0Ifeatures/clarity_embedding/clarity_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
>features/clarity_embedding/clarity_embedding_weights/Reshape_2Reshape=features/clarity_embedding/clarity_embedding_weights:output:0Dfeatures/clarity_embedding/clarity_embedding_weights/concat:output:0*
T0*'
_output_shapes
:??????????
 features/clarity_embedding/ShapeShapeGfeatures/clarity_embedding/clarity_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:x
.features/clarity_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0features/clarity_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0features/clarity_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(features/clarity_embedding/strided_sliceStridedSlice)features/clarity_embedding/Shape:output:07features/clarity_embedding/strided_slice/stack:output:09features/clarity_embedding/strided_slice/stack_1:output:09features/clarity_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*features/clarity_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
(features/clarity_embedding/Reshape/shapePack1features/clarity_embedding/strided_slice:output:03features/clarity_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
"features/clarity_embedding/ReshapeReshapeGfeatures/clarity_embedding/clarity_embedding_weights/Reshape_2:output:01features/clarity_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????r
'features/color_embedding/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
#features/color_embedding/ExpandDims
ExpandDimsinputs_color0features/color_embedding/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:??????????
7features/color_embedding/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB :
??????????
5features/color_embedding/to_sparse_input/ignore_valueCast@features/color_embedding/to_sparse_input/ignore_value/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
1features/color_embedding/to_sparse_input/NotEqualNotEqual,features/color_embedding/ExpandDims:output:09features/color_embedding/to_sparse_input/ignore_value:y:0*
T0	*'
_output_shapes
:??????????
0features/color_embedding/to_sparse_input/indicesWhere5features/color_embedding/to_sparse_input/NotEqual:z:0*'
_output_shapes
:??????????
/features/color_embedding/to_sparse_input/valuesGatherNd,features/color_embedding/ExpandDims:output:08features/color_embedding/to_sparse_input/indices:index:0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
4features/color_embedding/to_sparse_input/dense_shapeShape,features/color_embedding/ExpandDims:output:0*
T0	*
_output_shapes
:*
out_type0	?
6features/color_embedding/None_Lookup/LookupTableFindV2LookupTableFindV2Cfeatures_color_embedding_none_lookup_lookuptablefindv2_table_handle8features/color_embedding/to_sparse_input/values:output:0Dfeatures_color_embedding_none_lookup_lookuptablefindv2_default_value*	
Tin0	*

Tout0	*#
_output_shapes
:??????????
'features/color_embedding/ReadVariableOpReadVariableOp0features_color_embedding_readvariableop_resource*
_output_shapes

:*
dtype0?
<features/color_embedding/color_embedding_weights/Slice/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
;features/color_embedding/color_embedding_weights/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
6features/color_embedding/color_embedding_weights/SliceSlice=features/color_embedding/to_sparse_input/dense_shape:output:0Efeatures/color_embedding/color_embedding_weights/Slice/begin:output:0Dfeatures/color_embedding/color_embedding_weights/Slice/size:output:0*
Index0*
T0	*
_output_shapes
:?
6features/color_embedding/color_embedding_weights/ConstConst*
_output_shapes
:*
dtype0*
valueB: ?
5features/color_embedding/color_embedding_weights/ProdProd?features/color_embedding/color_embedding_weights/Slice:output:0?features/color_embedding/color_embedding_weights/Const:output:0*
T0	*
_output_shapes
: ?
Afeatures/color_embedding/color_embedding_weights/GatherV2/indicesConst*
_output_shapes
: *
dtype0*
value	B :?
>features/color_embedding/color_embedding_weights/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
9features/color_embedding/color_embedding_weights/GatherV2GatherV2=features/color_embedding/to_sparse_input/dense_shape:output:0Jfeatures/color_embedding/color_embedding_weights/GatherV2/indices:output:0Gfeatures/color_embedding/color_embedding_weights/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: ?
7features/color_embedding/color_embedding_weights/Cast/xPack>features/color_embedding/color_embedding_weights/Prod:output:0Bfeatures/color_embedding/color_embedding_weights/GatherV2:output:0*
N*
T0	*
_output_shapes
:?
>features/color_embedding/color_embedding_weights/SparseReshapeSparseReshape8features/color_embedding/to_sparse_input/indices:index:0=features/color_embedding/to_sparse_input/dense_shape:output:0@features/color_embedding/color_embedding_weights/Cast/x:output:0*-
_output_shapes
:?????????:?
Gfeatures/color_embedding/color_embedding_weights/SparseReshape/IdentityIdentity?features/color_embedding/None_Lookup/LookupTableFindV2:values:0*
T0	*#
_output_shapes
:??????????
?features/color_embedding/color_embedding_weights/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
=features/color_embedding/color_embedding_weights/GreaterEqualGreaterEqualPfeatures/color_embedding/color_embedding_weights/SparseReshape/Identity:output:0Hfeatures/color_embedding/color_embedding_weights/GreaterEqual/y:output:0*
T0	*#
_output_shapes
:??????????
6features/color_embedding/color_embedding_weights/WhereWhereAfeatures/color_embedding/color_embedding_weights/GreaterEqual:z:0*'
_output_shapes
:??????????
>features/color_embedding/color_embedding_weights/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
8features/color_embedding/color_embedding_weights/ReshapeReshape>features/color_embedding/color_embedding_weights/Where:index:0Gfeatures/color_embedding/color_embedding_weights/Reshape/shape:output:0*
T0	*#
_output_shapes
:??????????
@features/color_embedding/color_embedding_weights/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
;features/color_embedding/color_embedding_weights/GatherV2_1GatherV2Ofeatures/color_embedding/color_embedding_weights/SparseReshape:output_indices:0Afeatures/color_embedding/color_embedding_weights/Reshape:output:0Ifeatures/color_embedding/color_embedding_weights/GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:??????????
@features/color_embedding/color_embedding_weights/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
;features/color_embedding/color_embedding_weights/GatherV2_2GatherV2Pfeatures/color_embedding/color_embedding_weights/SparseReshape/Identity:output:0Afeatures/color_embedding/color_embedding_weights/Reshape:output:0Ifeatures/color_embedding/color_embedding_weights/GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:??????????
9features/color_embedding/color_embedding_weights/IdentityIdentityMfeatures/color_embedding/color_embedding_weights/SparseReshape:output_shape:0*
T0	*
_output_shapes
:?
Jfeatures/color_embedding/color_embedding_weights/SparseFillEmptyRows/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Xfeatures/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsDfeatures/color_embedding/color_embedding_weights/GatherV2_1:output:0Dfeatures/color_embedding/color_embedding_weights/GatherV2_2:output:0Bfeatures/color_embedding/color_embedding_weights/Identity:output:0Sfeatures/color_embedding/color_embedding_weights/SparseFillEmptyRows/Const:output:0*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:??????????
\features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        ?
^features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
^features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ?
Vfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSliceifeatures/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_indices:0efeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack:output:0gfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1:output:0gfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask?
Ofeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/UniqueUniquehfeatures/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:output_values:0*
T0	*2
_output_shapes 
:?????????:??????????
^features/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axisConst*:
_class0
.,loc:@features/color_embedding/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : ?
Yfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookupGatherV2/features/color_embedding/ReadVariableOp:value:0Sfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:y:0gfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*:
_class0
.,loc:@features/color_embedding/ReadVariableOp*'
_output_shapes
:??????????
bfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentitybfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup:output:0*
T0*'
_output_shapes
:??????????
Hfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparseSparseSegmentMeankfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity:output:0Ufeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse/Unique:idx:0_features/color_embedding/color_embedding_weights/embedding_lookup_sparse/strided_slice:output:0*
T0*
Tsegmentids0	*'
_output_shapes
:??????????
@features/color_embedding/color_embedding_weights/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
:features/color_embedding/color_embedding_weights/Reshape_1Reshapenfeatures/color_embedding/color_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:empty_row_indicator:0Ifeatures/color_embedding/color_embedding_weights/Reshape_1/shape:output:0*
T0
*'
_output_shapes
:??????????
6features/color_embedding/color_embedding_weights/ShapeShapeQfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*
_output_shapes
:?
Dfeatures/color_embedding/color_embedding_weights/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:?
Ffeatures/color_embedding/color_embedding_weights/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Ffeatures/color_embedding/color_embedding_weights/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
>features/color_embedding/color_embedding_weights/strided_sliceStridedSlice?features/color_embedding/color_embedding_weights/Shape:output:0Mfeatures/color_embedding/color_embedding_weights/strided_slice/stack:output:0Ofeatures/color_embedding/color_embedding_weights/strided_slice/stack_1:output:0Ofeatures/color_embedding/color_embedding_weights/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
8features/color_embedding/color_embedding_weights/stack/0Const*
_output_shapes
: *
dtype0*
value	B :?
6features/color_embedding/color_embedding_weights/stackPackAfeatures/color_embedding/color_embedding_weights/stack/0:output:0Gfeatures/color_embedding/color_embedding_weights/strided_slice:output:0*
N*
T0*
_output_shapes
:?
5features/color_embedding/color_embedding_weights/TileTileCfeatures/color_embedding/color_embedding_weights/Reshape_1:output:0?features/color_embedding/color_embedding_weights/stack:output:0*
T0
*'
_output_shapes
:??????????
;features/color_embedding/color_embedding_weights/zeros_like	ZerosLikeQfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
0features/color_embedding/color_embedding_weightsSelect>features/color_embedding/color_embedding_weights/Tile:output:0?features/color_embedding/color_embedding_weights/zeros_like:y:0Qfeatures/color_embedding/color_embedding_weights/embedding_lookup_sparse:output:0*
T0*'
_output_shapes
:??????????
7features/color_embedding/color_embedding_weights/Cast_1Cast=features/color_embedding/to_sparse_input/dense_shape:output:0*

DstT0*

SrcT0	*
_output_shapes
:?
>features/color_embedding/color_embedding_weights/Slice_1/beginConst*
_output_shapes
:*
dtype0*
valueB: ?
=features/color_embedding/color_embedding_weights/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:?
8features/color_embedding/color_embedding_weights/Slice_1Slice;features/color_embedding/color_embedding_weights/Cast_1:y:0Gfeatures/color_embedding/color_embedding_weights/Slice_1/begin:output:0Ffeatures/color_embedding/color_embedding_weights/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:?
8features/color_embedding/color_embedding_weights/Shape_1Shape9features/color_embedding/color_embedding_weights:output:0*
T0*
_output_shapes
:?
>features/color_embedding/color_embedding_weights/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB:?
=features/color_embedding/color_embedding_weights/Slice_2/sizeConst*
_output_shapes
:*
dtype0*
valueB:
??????????
8features/color_embedding/color_embedding_weights/Slice_2SliceAfeatures/color_embedding/color_embedding_weights/Shape_1:output:0Gfeatures/color_embedding/color_embedding_weights/Slice_2/begin:output:0Ffeatures/color_embedding/color_embedding_weights/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:~
<features/color_embedding/color_embedding_weights/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
7features/color_embedding/color_embedding_weights/concatConcatV2Afeatures/color_embedding/color_embedding_weights/Slice_1:output:0Afeatures/color_embedding/color_embedding_weights/Slice_2:output:0Efeatures/color_embedding/color_embedding_weights/concat/axis:output:0*
N*
T0*
_output_shapes
:?
:features/color_embedding/color_embedding_weights/Reshape_2Reshape9features/color_embedding/color_embedding_weights:output:0@features/color_embedding/color_embedding_weights/concat:output:0*
T0*'
_output_shapes
:??????????
features/color_embedding/ShapeShapeCfeatures/color_embedding/color_embedding_weights/Reshape_2:output:0*
T0*
_output_shapes
:v
,features/color_embedding/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.features/color_embedding/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.features/color_embedding/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&features/color_embedding/strided_sliceStridedSlice'features/color_embedding/Shape:output:05features/color_embedding/strided_slice/stack:output:07features/color_embedding/strided_slice/stack_1:output:07features/color_embedding/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(features/color_embedding/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :?
&features/color_embedding/Reshape/shapePack/features/color_embedding/strided_slice:output:01features/color_embedding/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:?
 features/color_embedding/ReshapeReshapeCfeatures/color_embedding/color_embedding_weights/Reshape_2:output:0/features/color_embedding/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????_
features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
??????????
features/concatConcatV2features/carat/Reshape:output:0+features/clarity_embedding/Reshape:output:0)features/color_embedding/Reshape:output:0features/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
 middlerelu/MatMul/ReadVariableOpReadVariableOp)middlerelu_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
middlerelu/MatMulMatMulfeatures/concat:output:0(middlerelu/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
!middlerelu/BiasAdd/ReadVariableOpReadVariableOp*middlerelu_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
middlerelu/BiasAddBiasAddmiddlerelu/MatMul:product:0)middlerelu/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????g
middlerelu/ReluRelumiddlerelu/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_2/MatMulMatMulmiddlerelu/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????b
IdentityIdentitydense_2/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp9^features/clarity_embedding/None_Lookup/LookupTableFindV2*^features/clarity_embedding/ReadVariableOp7^features/color_embedding/None_Lookup/LookupTableFindV2(^features/color_embedding/ReadVariableOp"^middlerelu/BiasAdd/ReadVariableOp!^middlerelu/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2t
8features/clarity_embedding/None_Lookup/LookupTableFindV28features/clarity_embedding/None_Lookup/LookupTableFindV22V
)features/clarity_embedding/ReadVariableOp)features/clarity_embedding/ReadVariableOp2p
6features/color_embedding/None_Lookup/LookupTableFindV26features/color_embedding/None_Lookup/LookupTableFindV22R
'features/color_embedding/ReadVariableOp'features/color_embedding/ReadVariableOp2F
!middlerelu/BiasAdd/ReadVariableOp!middlerelu/BiasAdd/ReadVariableOp2D
 middlerelu/MatMul/ReadVariableOp middlerelu/MatMul/ReadVariableOp:Q M
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
?
-
__inference__destroyer_384590
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
?
?
-__inference_sequential_2_layer_call_fn_383142	
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
GPU 2J 8? *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_383119o
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
??
?
D__inference_features_layer_call_and_return_conditional_losses_383070
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
?
-
__inference__destroyer_384608
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
;
__inference__creator_384577
identity??
hash_tablen

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name360947*
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
?

?
F__inference_middlerelu_layer_call_and_return_conditional_losses_383095

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
?
?
H__inference_sequential_2_layer_call_and_return_conditional_losses_383603	
carat
clarity		
color	
cut		
depth	
table
x
y
z
features_383579
features_383581	!
features_383583:
features_383585
features_383587	!
features_383589:$
middlerelu_383592:	? 
middlerelu_383594:	?!
dense_2_383597:	?
dense_2_383599:
identity??dense_2/StatefulPartitionedCall? features/StatefulPartitionedCall?"middlerelu/StatefulPartitionedCall?
 features/StatefulPartitionedCallStatefulPartitionedCallcaratclaritycolorcutdepthtablexyzfeatures_383579features_383581features_383583features_383585features_383587features_383589*
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
GPU 2J 8? *M
fHRF
D__inference_features_layer_call_and_return_conditional_losses_383384?
"middlerelu/StatefulPartitionedCallStatefulPartitionedCall)features/StatefulPartitionedCall:output:0middlerelu_383592middlerelu_383594*
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
GPU 2J 8? *O
fJRH
F__inference_middlerelu_layer_call_and_return_conditional_losses_383095?
dense_2/StatefulPartitionedCallStatefulPartitionedCall+middlerelu/StatefulPartitionedCall:output:0dense_2_383597dense_2_383599*
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
GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_383112w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall!^features/StatefulPartitionedCall#^middlerelu/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2D
 features/StatefulPartitionedCall features/StatefulPartitionedCall2H
"middlerelu/StatefulPartitionedCall"middlerelu/StatefulPartitionedCall:J F
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
-__inference_sequential_2_layer_call_fn_383642
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
GPU 2J 8? *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_383119o
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
C__inference_dense_2_layer_call_and_return_conditional_losses_383112

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
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
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
?
?
__inference_<lambda>_3846165
1table_init360946_lookuptableimportv2_table_handle-
)table_init360946_lookuptableimportv2_keys	/
+table_init360946_lookuptableimportv2_values	
identity??$table_init360946/LookupTableImportV2?
$table_init360946/LookupTableImportV2LookupTableImportV21table_init360946_lookuptableimportv2_table_handle)table_init360946_lookuptableimportv2_keys+table_init360946_lookuptableimportv2_values*	
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
: m
NoOpNoOp%^table_init360946/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2L
$table_init360946/LookupTableImportV2$table_init360946/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
??
?
D__inference_features_layer_call_and_return_conditional_losses_384532
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
??
?
"__inference__traced_restore_384885
file_prefix\
Jassignvariableop_sequential_2_features_clarity_embedding_embedding_weights:\
Jassignvariableop_1_sequential_2_features_color_embedding_embedding_weights:D
1assignvariableop_2_sequential_2_middlerelu_kernel:	?>
/assignvariableop_3_sequential_2_middlerelu_bias:	?A
.assignvariableop_4_sequential_2_dense_2_kernel:	?:
,assignvariableop_5_sequential_2_dense_2_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: 0
"assignvariableop_15_true_positives:1
#assignvariableop_16_false_positives:2
$assignvariableop_17_true_positives_1:1
#assignvariableop_18_false_negatives:2
$assignvariableop_19_true_positives_2:d0
"assignvariableop_20_true_negatives:d3
%assignvariableop_21_false_positives_1:d3
%assignvariableop_22_false_negatives_1:df
Tassignvariableop_23_adam_sequential_2_features_clarity_embedding_embedding_weights_m:d
Rassignvariableop_24_adam_sequential_2_features_color_embedding_embedding_weights_m:L
9assignvariableop_25_adam_sequential_2_middlerelu_kernel_m:	?F
7assignvariableop_26_adam_sequential_2_middlerelu_bias_m:	?I
6assignvariableop_27_adam_sequential_2_dense_2_kernel_m:	?B
4assignvariableop_28_adam_sequential_2_dense_2_bias_m:f
Tassignvariableop_29_adam_sequential_2_features_clarity_embedding_embedding_weights_v:d
Rassignvariableop_30_adam_sequential_2_features_color_embedding_embedding_weights_v:L
9assignvariableop_31_adam_sequential_2_middlerelu_kernel_v:	?F
7assignvariableop_32_adam_sequential_2_middlerelu_bias_v:	?I
6assignvariableop_33_adam_sequential_2_dense_2_kernel_v:	?B
4assignvariableop_34_adam_sequential_2_dense_2_bias_v:
identity_36??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$BTlayer_with_weights-0/clarity_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/color_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/clarity_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBnlayer_with_weights-0/color_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/clarity_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBnlayer_with_weights-0/color_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpJassignvariableop_sequential_2_features_clarity_embedding_embedding_weightsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpJassignvariableop_1_sequential_2_features_color_embedding_embedding_weightsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp1assignvariableop_2_sequential_2_middlerelu_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp/assignvariableop_3_sequential_2_middlerelu_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp.assignvariableop_4_sequential_2_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp,assignvariableop_5_sequential_2_dense_2_biasIdentity_5:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp"assignvariableop_15_true_positivesIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_false_positivesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp$assignvariableop_17_true_positives_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp#assignvariableop_18_false_negativesIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_true_positives_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_true_negativesIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp%assignvariableop_21_false_positives_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp%assignvariableop_22_false_negatives_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpTassignvariableop_23_adam_sequential_2_features_clarity_embedding_embedding_weights_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpRassignvariableop_24_adam_sequential_2_features_color_embedding_embedding_weights_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp9assignvariableop_25_adam_sequential_2_middlerelu_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp7assignvariableop_26_adam_sequential_2_middlerelu_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_sequential_2_dense_2_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp4assignvariableop_28_adam_sequential_2_dense_2_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpTassignvariableop_29_adam_sequential_2_features_clarity_embedding_embedding_weights_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpRassignvariableop_30_adam_sequential_2_features_color_embedding_embedding_weights_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp9assignvariableop_31_adam_sequential_2_middlerelu_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp7assignvariableop_32_adam_sequential_2_middlerelu_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adam_sequential_2_dense_2_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_sequential_2_dense_2_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_36IdentityIdentity_35:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_36Identity_36:output:0*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
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
?
?
$__inference_signature_wrapper_384110	
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
GPU 2J 8? **
f%R#
!__inference__wrapped_model_382861o
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
C__inference_dense_2_layer_call_and_return_conditional_losses_384572

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
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
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
?
?
-__inference_sequential_2_layer_call_fn_383675
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
GPU 2J 8? *Q
fLRJ
H__inference_sequential_2_layer_call_and_return_conditional_losses_383477o
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
?P
?
__inference__traced_save_384770
file_prefixX
Tsavev2_sequential_2_features_clarity_embedding_embedding_weights_read_readvariableopV
Rsavev2_sequential_2_features_color_embedding_embedding_weights_read_readvariableop=
9savev2_sequential_2_middlerelu_kernel_read_readvariableop;
7savev2_sequential_2_middlerelu_bias_read_readvariableop:
6savev2_sequential_2_dense_2_kernel_read_readvariableop8
4savev2_sequential_2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_negatives_read_readvariableop/
+savev2_true_positives_2_read_readvariableop-
)savev2_true_negatives_read_readvariableop0
,savev2_false_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop_
[savev2_adam_sequential_2_features_clarity_embedding_embedding_weights_m_read_readvariableop]
Ysavev2_adam_sequential_2_features_color_embedding_embedding_weights_m_read_readvariableopD
@savev2_adam_sequential_2_middlerelu_kernel_m_read_readvariableopB
>savev2_adam_sequential_2_middlerelu_bias_m_read_readvariableopA
=savev2_adam_sequential_2_dense_2_kernel_m_read_readvariableop?
;savev2_adam_sequential_2_dense_2_bias_m_read_readvariableop_
[savev2_adam_sequential_2_features_clarity_embedding_embedding_weights_v_read_readvariableop]
Ysavev2_adam_sequential_2_features_color_embedding_embedding_weights_v_read_readvariableopD
@savev2_adam_sequential_2_middlerelu_kernel_v_read_readvariableopB
>savev2_adam_sequential_2_middlerelu_bias_v_read_readvariableopA
=savev2_adam_sequential_2_dense_2_kernel_v_read_readvariableop?
;savev2_adam_sequential_2_dense_2_bias_v_read_readvariableop
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
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$BTlayer_with_weights-0/clarity_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/color_embedding.Sembedding_weights/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/4/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/4/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/clarity_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBnlayer_with_weights-0/color_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBplayer_with_weights-0/clarity_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBnlayer_with_weights-0/color_embedding.Sembedding_weights/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Tsavev2_sequential_2_features_clarity_embedding_embedding_weights_read_readvariableopRsavev2_sequential_2_features_color_embedding_embedding_weights_read_readvariableop9savev2_sequential_2_middlerelu_kernel_read_readvariableop7savev2_sequential_2_middlerelu_bias_read_readvariableop6savev2_sequential_2_dense_2_kernel_read_readvariableop4savev2_sequential_2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_2_read_readvariableop)savev2_true_negatives_read_readvariableop,savev2_false_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableop[savev2_adam_sequential_2_features_clarity_embedding_embedding_weights_m_read_readvariableopYsavev2_adam_sequential_2_features_color_embedding_embedding_weights_m_read_readvariableop@savev2_adam_sequential_2_middlerelu_kernel_m_read_readvariableop>savev2_adam_sequential_2_middlerelu_bias_m_read_readvariableop=savev2_adam_sequential_2_dense_2_kernel_m_read_readvariableop;savev2_adam_sequential_2_dense_2_bias_m_read_readvariableop[savev2_adam_sequential_2_features_clarity_embedding_embedding_weights_v_read_readvariableopYsavev2_adam_sequential_2_features_color_embedding_embedding_weights_v_read_readvariableop@savev2_adam_sequential_2_middlerelu_kernel_v_read_readvariableop>savev2_adam_sequential_2_middlerelu_bias_v_read_readvariableop=savev2_adam_sequential_2_dense_2_kernel_v_read_readvariableop;savev2_adam_sequential_2_dense_2_bias_v_read_readvariableopsavev2_const_6"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$	?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :::	?:?:	?:: : : : : : : : : :::::d:d:d:d:::	?:?:	?::::	?:?:	?:: 2(
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
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:d: 

_output_shapes
:d: 

_output_shapes
:d: 

_output_shapes
:d:$ 

_output_shapes

::$ 

_output_shapes

::%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

::% !

_output_shapes
:	?:!!

_output_shapes	
:?:%"!

_output_shapes
:	?: #

_output_shapes
::$

_output_shapes
: 
??
?
D__inference_features_layer_call_and_return_conditional_losses_384346
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
?
?
)__inference_features_layer_call_fn_384135
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
GPU 2J 8? *M
fHRF
D__inference_features_layer_call_and_return_conditional_losses_383070o
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
?
?
H__inference_sequential_2_layer_call_and_return_conditional_losses_383477

inputs
inputs_1	
inputs_2	
inputs_3	
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
features_383453
features_383455	!
features_383457:
features_383459
features_383461	!
features_383463:$
middlerelu_383466:	? 
middlerelu_383468:	?!
dense_2_383471:	?
dense_2_383473:
identity??dense_2/StatefulPartitionedCall? features/StatefulPartitionedCall?"middlerelu/StatefulPartitionedCall?
 features/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8features_383453features_383455features_383457features_383459features_383461features_383463*
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
GPU 2J 8? *M
fHRF
D__inference_features_layer_call_and_return_conditional_losses_383384?
"middlerelu/StatefulPartitionedCallStatefulPartitionedCall)features/StatefulPartitionedCall:output:0middlerelu_383466middlerelu_383468*
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
GPU 2J 8? *O
fJRH
F__inference_middlerelu_layer_call_and_return_conditional_losses_383095?
dense_2/StatefulPartitionedCallStatefulPartitionedCall+middlerelu/StatefulPartitionedCall:output:0dense_2_383471dense_2_383473*
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
GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_383112w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall!^features/StatefulPartitionedCall#^middlerelu/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2D
 features/StatefulPartitionedCall features/StatefulPartitionedCall2H
"middlerelu/StatefulPartitionedCall"middlerelu/StatefulPartitionedCall:K G
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
?
?
+__inference_middlerelu_layer_call_fn_384541

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
GPU 2J 8? *O
fJRH
F__inference_middlerelu_layer_call_and_return_conditional_losses_383095p
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
?
?
H__inference_sequential_2_layer_call_and_return_conditional_losses_383568	
carat
clarity		
color	
cut		
depth	
table
x
y
z
features_383544
features_383546	!
features_383548:
features_383550
features_383552	!
features_383554:$
middlerelu_383557:	? 
middlerelu_383559:	?!
dense_2_383562:	?
dense_2_383564:
identity??dense_2/StatefulPartitionedCall? features/StatefulPartitionedCall?"middlerelu/StatefulPartitionedCall?
 features/StatefulPartitionedCallStatefulPartitionedCallcaratclaritycolorcutdepthtablexyzfeatures_383544features_383546features_383548features_383550features_383552features_383554*
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
GPU 2J 8? *M
fHRF
D__inference_features_layer_call_and_return_conditional_losses_383070?
"middlerelu/StatefulPartitionedCallStatefulPartitionedCall)features/StatefulPartitionedCall:output:0middlerelu_383557middlerelu_383559*
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
GPU 2J 8? *O
fJRH
F__inference_middlerelu_layer_call_and_return_conditional_losses_383095?
dense_2/StatefulPartitionedCallStatefulPartitionedCall+middlerelu/StatefulPartitionedCall:output:0dense_2_383562dense_2_383564*
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
GPU 2J 8? *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_383112w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_2/StatefulPartitionedCall!^features/StatefulPartitionedCall#^middlerelu/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : : : 2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2D
 features/StatefulPartitionedCall features/StatefulPartitionedCall2H
"middlerelu/StatefulPartitionedCall"middlerelu/StatefulPartitionedCall:J F
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
__inference_<lambda>_3846245
1table_init361038_lookuptableimportv2_table_handle-
)table_init361038_lookuptableimportv2_keys	/
+table_init361038_lookuptableimportv2_values	
identity??$table_init361038/LookupTableImportV2?
$table_init361038/LookupTableImportV2LookupTableImportV21table_init361038_lookuptableimportv2_table_handle)table_init361038_lookuptableimportv2_keys+table_init361038_lookuptableimportv2_values*	
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
: m
NoOpNoOp%^table_init361038/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2L
$table_init361038/LookupTableImportV2$table_init361038/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
;
__inference__creator_384595
identity??
hash_tablen

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*
shared_name361039*
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
__inference__initializer_3845855
1table_init360946_lookuptableimportv2_table_handle-
)table_init360946_lookuptableimportv2_keys	/
+table_init360946_lookuptableimportv2_values	
identity??$table_init360946/LookupTableImportV2?
$table_init360946/LookupTableImportV2LookupTableImportV21table_init360946_lookuptableimportv2_table_handle)table_init360946_lookuptableimportv2_keys+table_init360946_lookuptableimportv2_values*	
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
: m
NoOpNoOp%^table_init360946/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2L
$table_init360946/LookupTableImportV2$table_init360946/LookupTableImportV2: 
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
,learning_ratemlmmmnmo mp!mqvrvsvtvu vv!vw"
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
?2?
-__inference_sequential_2_layer_call_fn_383142
-__inference_sequential_2_layer_call_fn_383642
-__inference_sequential_2_layer_call_fn_383675
-__inference_sequential_2_layer_call_fn_383533?
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
H__inference_sequential_2_layer_call_and_return_conditional_losses_383875
H__inference_sequential_2_layer_call_and_return_conditional_losses_384075
H__inference_sequential_2_layer_call_and_return_conditional_losses_383568
H__inference_sequential_2_layer_call_and_return_conditional_losses_383603?
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
!__inference__wrapped_model_382861caratclaritycolorcutdepthtablexyz	"?
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
K:I29sequential_2/features/clarity_embedding/embedding_weights
I:G27sequential_2/features/color_embedding/embedding_weights
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
)__inference_features_layer_call_fn_384135
)__inference_features_layer_call_fn_384160?
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
D__inference_features_layer_call_and_return_conditional_losses_384346
D__inference_features_layer_call_and_return_conditional_losses_384532?
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
1:/	?2sequential_2/middlerelu/kernel
+:)?2sequential_2/middlerelu/bias
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
+__inference_middlerelu_layer_call_fn_384541?
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
F__inference_middlerelu_layer_call_and_return_conditional_losses_384552?
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
.:,	?2sequential_2/dense_2/kernel
':%2sequential_2/dense_2/bias
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
(__inference_dense_2_layer_call_fn_384561?
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
C__inference_dense_2_layer_call_and_return_conditional_losses_384572?
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
C
D0
E1
F2
G3
H4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
$__inference_signature_wrapper_384110caratclaritycolorcutdepthtablexyz"?
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
Iclarity_lookup"
_generic_user_object
0
Jcolor_lookup"
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
	Ktotal
	Lcount
M	variables
N	keras_api"
_tf_keras_metric
^
	Ototal
	Pcount
Q
_fn_kwargs
R	variables
S	keras_api"
_tf_keras_metric
q
T
thresholds
Utrue_positives
Vfalse_positives
W	variables
X	keras_api"
_tf_keras_metric
q
Y
thresholds
Ztrue_positives
[false_negatives
\	variables
]	keras_api"
_tf_keras_metric
?
^true_positives
_true_negatives
`false_positives
afalse_negatives
b	variables
c	keras_api"
_tf_keras_metric
j
d_initializer
e_create_resource
f_initialize
g_destroy_resourceR jCustom.StaticHashTable
j
h_initializer
i_create_resource
j_initialize
k_destroy_resourceR jCustom.StaticHashTable
:  (2total
:  (2count
.
K0
L1"
trackable_list_wrapper
-
M	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
O0
P1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
.
U0
V1"
trackable_list_wrapper
-
W	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
.
Z0
[1"
trackable_list_wrapper
-
\	variables"
_generic_user_object
:d (2true_positives
:d (2true_negatives
:d (2false_positives
:d (2false_negatives
<
^0
_1
`2
a3"
trackable_list_wrapper
-
b	variables"
_generic_user_object
"
_generic_user_object
?2?
__inference__creator_384577?
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
__inference__initializer_384585?
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
__inference__destroyer_384590?
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
__inference__creator_384595?
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
__inference__initializer_384603?
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
__inference__destroyer_384608?
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
P:N2@Adam/sequential_2/features/clarity_embedding/embedding_weights/m
N:L2>Adam/sequential_2/features/color_embedding/embedding_weights/m
6:4	?2%Adam/sequential_2/middlerelu/kernel/m
0:.?2#Adam/sequential_2/middlerelu/bias/m
3:1	?2"Adam/sequential_2/dense_2/kernel/m
,:*2 Adam/sequential_2/dense_2/bias/m
P:N2@Adam/sequential_2/features/clarity_embedding/embedding_weights/v
N:L2>Adam/sequential_2/features/color_embedding/embedding_weights/v
6:4	?2%Adam/sequential_2/middlerelu/kernel/v
0:.?2#Adam/sequential_2/middlerelu/bias/v
3:1	?2"Adam/sequential_2/dense_2/kernel/v
,:*2 Adam/sequential_2/dense_2/bias/v
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
Const_57
__inference__creator_384577?

? 
? "? 7
__inference__creator_384595?

? 
? "? 9
__inference__destroyer_384590?

? 
? "? 9
__inference__destroyer_384608?

? 
? "? @
__inference__initializer_384585Iz{?

? 
? "? @
__inference__initializer_384603J|}?

? 
? "? ?
!__inference__wrapped_model_382861?
IxJy !???
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
C__inference_dense_2_layer_call_and_return_conditional_losses_384572] !0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_dense_2_layer_call_fn_384561P !0?-
&?#
!?
inputs??????????
? "???????????
D__inference_features_layer_call_and_return_conditional_losses_384346?IxJy???
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
D__inference_features_layer_call_and_return_conditional_losses_384532?IxJy???
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
)__inference_features_layer_call_fn_384135?IxJy???
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
)__inference_features_layer_call_fn_384160?IxJy???
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
? "???????????
F__inference_middlerelu_layer_call_and_return_conditional_losses_384552]/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? 
+__inference_middlerelu_layer_call_fn_384541P/?,
%?"
 ?
inputs?????????
? "????????????
H__inference_sequential_2_layer_call_and_return_conditional_losses_383568?
IxJy !???
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
H__inference_sequential_2_layer_call_and_return_conditional_losses_383603?
IxJy !???
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
H__inference_sequential_2_layer_call_and_return_conditional_losses_383875?
IxJy !???
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
H__inference_sequential_2_layer_call_and_return_conditional_losses_384075?
IxJy !???
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
-__inference_sequential_2_layer_call_fn_383142?
IxJy !???
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
-__inference_sequential_2_layer_call_fn_383533?
IxJy !???
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
-__inference_sequential_2_layer_call_fn_383642?
IxJy !???
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
-__inference_sequential_2_layer_call_fn_383675?
IxJy !???
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
$__inference_signature_wrapper_384110?
IxJy !???
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