
H
input/locationPlaceholder*
dtype0*
shape:���������
6
input/phasePlaceholder*
dtype0
*
shape:
7
input/ble_locationIdentityinput/location*
T0
�
ble_location/log_sigma2_0Const*�
value�B�"���j������o����3���� �������������(����6��H������r��#����n����Z��d�����9����6��T������ȑ������������r���� ��N������*�����*
dtype0
|
ble_location/log_sigma2_0/readIdentityble_location/log_sigma2_0*
T0*,
_class"
 loc:@ble_location/log_sigma2_0
�
ble_location/w_0Const*�
value�B�"�BA�8���7�����3 ?e%�7-�S:~�9��:`���58��7�q�E>��P�R�>Wo 6�q������/ɽ��>������_=b��>/�n:�ۗ:C/��C���
0�=x��	k=�|�:=ݏ����8��%��.�>v���)�8�!�>`@�9[�#>�ˬ>0O�9,̦=0�>���>U#�>���=���>*
dtype0
a
ble_location/w_0/readIdentityble_location/w_0*
T0*#
_class
loc:@ble_location/w_0
}
ble_location/b_0Const*U
valueLBJ"@N;���!������1U=:��<��7<8�;m��<B���W�5�H؜���~���9=�&����i="孼*
dtype0
a
ble_location/b_0/readIdentityble_location/b_0*
T0*#
_class
loc:@ble_location/b_0
=
ble_location/SquareSquareble_location/w_0/read*
T0
?
ble_location/add/yConst*
valueB
 *w�+2*
dtype0
I
ble_location/addAddble_location/Squareble_location/add/y*
T0
2
ble_location/LogLogble_location/add*
T0
R
ble_location/subSubble_location/log_sigma2_0/readble_location/Log*
T0
Q
$ble_location/clip_by_value/Minimum/yConst*
valueB
 *   A*
dtype0
n
"ble_location/clip_by_value/MinimumMinimumble_location/sub$ble_location/clip_by_value/Minimum/y*
T0
I
ble_location/clip_by_value/yConst*
valueB
 *   �*
dtype0
p
ble_location/clip_by_valueMaximum"ble_location/clip_by_value/Minimumble_location/clip_by_value/y*
T0
I
ble_location/log_alpha_0Identityble_location/clip_by_value*
T0
@
ble_location/Less/yConst*
valueB
 *  @@*
dtype0
Q
ble_location/LessLessble_location/log_alpha_0ble_location/Less/y*
T0
D
ble_location/CastCastble_location/Less*

SrcT0
*

DstT0
E
ble_location/cond/SwitchSwitchinput/phaseinput/phase*
T0

K
ble_location/cond/switch_tIdentityble_location/cond/Switch:1*
T0

;
ble_location/cond/pred_idIdentityinput/phase*
T0

�
ble_location/cond/MatMul/SwitchSwitchinput/ble_locationble_location/cond/pred_id*
T0*%
_class
loc:@input/ble_location
�
!ble_location/cond/MatMul/Switch_1Switchble_location/w_0/readble_location/cond/pred_id*
T0*#
_class
loc:@ble_location/w_0
�
ble_location/cond/MatMulMatMul!ble_location/cond/MatMul/Switch:1#ble_location/cond/MatMul/Switch_1:1*
T0*
transpose_a( *
transpose_b( 
N
ble_location/cond/SquareSquare!ble_location/cond/MatMul/Switch:1*
T0
�
ble_location/cond/Exp/SwitchSwitchble_location/log_alpha_0ble_location/cond/pred_id*
T0*+
_class!
loc:@ble_location/log_alpha_0
E
ble_location/cond/ExpExpble_location/cond/Exp/Switch:1*
T0
R
ble_location/cond/Square_1Square#ble_location/cond/MatMul/Switch_1:1*
T0
X
ble_location/cond/mulMulble_location/cond/Expble_location/cond/Square_1*
T0
�
ble_location/cond/MatMul_1MatMulble_location/cond/Squareble_location/cond/mul*
transpose_a( *
transpose_b( *
T0
a
ble_location/cond/add/yConst^ble_location/cond/switch_t*
valueB
 *w�+2*
dtype0
Z
ble_location/cond/addAddble_location/cond/MatMul_1ble_location/cond/add/y*
T0
>
ble_location/cond/SqrtSqrtble_location/cond/add*
T0
S
ble_location/cond/ShapeShapeble_location/cond/MatMul*
T0*
out_type0
n
$ble_location/cond/random_normal/meanConst^ble_location/cond/switch_t*
valueB
 *    *
dtype0
p
&ble_location/cond/random_normal/stddevConst^ble_location/cond/switch_t*
valueB
 *  �?*
dtype0
�
4ble_location/cond/random_normal/RandomStandardNormalRandomStandardNormalble_location/cond/Shape*
T0*
dtype0*
seed2 *

seed 
�
#ble_location/cond/random_normal/mulMul4ble_location/cond/random_normal/RandomStandardNormal&ble_location/cond/random_normal/stddev*
T0
z
ble_location/cond/random_normalAdd#ble_location/cond/random_normal/mul$ble_location/cond/random_normal/mean*
T0
`
ble_location/cond/mul_1Mulble_location/cond/Sqrtble_location/cond/random_normal*
T0
Z
ble_location/cond/add_1Addble_location/cond/MatMulble_location/cond/mul_1*
T0
�
ble_location/cond/mul_2/SwitchSwitchble_location/w_0/readble_location/cond/pred_id*
T0*#
_class
loc:@ble_location/w_0
�
 ble_location/cond/mul_2/Switch_1Switchble_location/Castble_location/cond/pred_id*
T0*$
_class
loc:@ble_location/Cast
i
ble_location/cond/mul_2Mulble_location/cond/mul_2/Switch ble_location/cond/mul_2/Switch_1*
T0
�
!ble_location/cond/MatMul_2/SwitchSwitchinput/ble_locationble_location/cond/pred_id*
T0*%
_class
loc:@input/ble_location
�
ble_location/cond/MatMul_2MatMul!ble_location/cond/MatMul_2/Switchble_location/cond/mul_2*
T0*
transpose_a( *
transpose_b( 
g
ble_location/cond/MergeMergeble_location/cond/MatMul_2ble_location/cond/add_1*
N*
T0
R
ble_location/add_1Addble_location/cond/Mergeble_location/b_0/read*
T0
6
ble_location/ReluReluble_location/add_1*
T0
�
+ble_location/ble_location/log_sigma2_outputConst*
dtype0*�
value�B�"�+��p���������
���������ґ�����"��
�����������������q�����������Α���������
�
0ble_location/ble_location/log_sigma2_output/readIdentity+ble_location/ble_location/log_sigma2_output*
T0*>
_class4
20loc:@ble_location/ble_location/log_sigma2_output
�
"ble_location/ble_location/w_outputConst*�
value�B�"��R������H%�GYD��)�O��mr�hz�>6�H��Zw9�<����.>��9�R��:6�.��6>F;�>����H��}7�6�ڹ�z��]�>$i_���>Q�g>B-�>򴟾�:>>D�>	>�慾*
dtype0
�
'ble_location/ble_location/w_output/readIdentity"ble_location/ble_location/w_output*
T0*5
_class+
)'loc:@ble_location/ble_location/w_output
W
"ble_location/ble_location/b_outputConst*
dtype0*
valueB"׏�#�=
�
'ble_location/ble_location/b_output/readIdentity"ble_location/ble_location/b_output*
T0*5
_class+
)'loc:@ble_location/ble_location/b_output
^
"ble_location_1/ble_location/SquareSquare'ble_location/ble_location/w_output/read*
T0
N
!ble_location_1/ble_location/add/yConst*
dtype0*
valueB
 *w�+2
v
ble_location_1/ble_location/addAdd"ble_location_1/ble_location/Square!ble_location_1/ble_location/add/y*
T0
P
ble_location_1/ble_location/LogLogble_location_1/ble_location/add*
T0
�
ble_location_1/ble_location/subSub0ble_location/ble_location/log_sigma2_output/readble_location_1/ble_location/Log*
T0
`
3ble_location_1/ble_location/clip_by_value/Minimum/yConst*
valueB
 *   A*
dtype0
�
1ble_location_1/ble_location/clip_by_value/MinimumMinimumble_location_1/ble_location/sub3ble_location_1/ble_location/clip_by_value/Minimum/y*
T0
X
+ble_location_1/ble_location/clip_by_value/yConst*
valueB
 *   �*
dtype0
�
)ble_location_1/ble_location/clip_by_valueMaximum1ble_location_1/ble_location/clip_by_value/Minimum+ble_location_1/ble_location/clip_by_value/y*
T0
l
,ble_location_1/ble_location/log_alpha_outputIdentity)ble_location_1/ble_location/clip_by_value*
T0
O
"ble_location_1/ble_location/Less/yConst*
valueB
 *  @@*
dtype0
�
 ble_location_1/ble_location/LessLess,ble_location_1/ble_location/log_alpha_output"ble_location_1/ble_location/Less/y*
T0
b
 ble_location_1/ble_location/CastCast ble_location_1/ble_location/Less*

SrcT0
*

DstT0
T
'ble_location_1/ble_location/cond/SwitchSwitchinput/phaseinput/phase*
T0

i
)ble_location_1/ble_location/cond/switch_tIdentity)ble_location_1/ble_location/cond/Switch:1*
T0

J
(ble_location_1/ble_location/cond/pred_idIdentityinput/phase*
T0

�
.ble_location_1/ble_location/cond/MatMul/SwitchSwitchble_location/Relu(ble_location_1/ble_location/cond/pred_id*
T0*$
_class
loc:@ble_location/Relu
�
0ble_location_1/ble_location/cond/MatMul/Switch_1Switch'ble_location/ble_location/w_output/read(ble_location_1/ble_location/cond/pred_id*
T0*5
_class+
)'loc:@ble_location/ble_location/w_output
�
'ble_location_1/ble_location/cond/MatMulMatMul0ble_location_1/ble_location/cond/MatMul/Switch:12ble_location_1/ble_location/cond/MatMul/Switch_1:1*
transpose_a( *
transpose_b( *
T0
l
'ble_location_1/ble_location/cond/SquareSquare0ble_location_1/ble_location/cond/MatMul/Switch:1*
T0
�
+ble_location_1/ble_location/cond/Exp/SwitchSwitch,ble_location_1/ble_location/log_alpha_output(ble_location_1/ble_location/cond/pred_id*
T0*?
_class5
31loc:@ble_location_1/ble_location/log_alpha_output
c
$ble_location_1/ble_location/cond/ExpExp-ble_location_1/ble_location/cond/Exp/Switch:1*
T0
p
)ble_location_1/ble_location/cond/Square_1Square2ble_location_1/ble_location/cond/MatMul/Switch_1:1*
T0
�
$ble_location_1/ble_location/cond/mulMul$ble_location_1/ble_location/cond/Exp)ble_location_1/ble_location/cond/Square_1*
T0
�
)ble_location_1/ble_location/cond/MatMul_1MatMul'ble_location_1/ble_location/cond/Square$ble_location_1/ble_location/cond/mul*
transpose_a( *
transpose_b( *
T0

&ble_location_1/ble_location/cond/add/yConst*^ble_location_1/ble_location/cond/switch_t*
dtype0*
valueB
 *w�+2
�
$ble_location_1/ble_location/cond/addAdd)ble_location_1/ble_location/cond/MatMul_1&ble_location_1/ble_location/cond/add/y*
T0
\
%ble_location_1/ble_location/cond/SqrtSqrt$ble_location_1/ble_location/cond/add*
T0
q
&ble_location_1/ble_location/cond/ShapeShape'ble_location_1/ble_location/cond/MatMul*
T0*
out_type0
�
3ble_location_1/ble_location/cond/random_normal/meanConst*^ble_location_1/ble_location/cond/switch_t*
valueB
 *    *
dtype0
�
5ble_location_1/ble_location/cond/random_normal/stddevConst*^ble_location_1/ble_location/cond/switch_t*
valueB
 *  �?*
dtype0
�
Cble_location_1/ble_location/cond/random_normal/RandomStandardNormalRandomStandardNormal&ble_location_1/ble_location/cond/Shape*

seed *
T0*
dtype0*
seed2 
�
2ble_location_1/ble_location/cond/random_normal/mulMulCble_location_1/ble_location/cond/random_normal/RandomStandardNormal5ble_location_1/ble_location/cond/random_normal/stddev*
T0
�
.ble_location_1/ble_location/cond/random_normalAdd2ble_location_1/ble_location/cond/random_normal/mul3ble_location_1/ble_location/cond/random_normal/mean*
T0
�
&ble_location_1/ble_location/cond/mul_1Mul%ble_location_1/ble_location/cond/Sqrt.ble_location_1/ble_location/cond/random_normal*
T0
�
&ble_location_1/ble_location/cond/add_1Add'ble_location_1/ble_location/cond/MatMul&ble_location_1/ble_location/cond/mul_1*
T0
�
-ble_location_1/ble_location/cond/mul_2/SwitchSwitch'ble_location/ble_location/w_output/read(ble_location_1/ble_location/cond/pred_id*
T0*5
_class+
)'loc:@ble_location/ble_location/w_output
�
/ble_location_1/ble_location/cond/mul_2/Switch_1Switch ble_location_1/ble_location/Cast(ble_location_1/ble_location/cond/pred_id*
T0*3
_class)
'%loc:@ble_location_1/ble_location/Cast
�
&ble_location_1/ble_location/cond/mul_2Mul-ble_location_1/ble_location/cond/mul_2/Switch/ble_location_1/ble_location/cond/mul_2/Switch_1*
T0
�
0ble_location_1/ble_location/cond/MatMul_2/SwitchSwitchble_location/Relu(ble_location_1/ble_location/cond/pred_id*
T0*$
_class
loc:@ble_location/Relu
�
)ble_location_1/ble_location/cond/MatMul_2MatMul0ble_location_1/ble_location/cond/MatMul_2/Switch&ble_location_1/ble_location/cond/mul_2*
transpose_b( *
T0*
transpose_a( 
�
&ble_location_1/ble_location/cond/MergeMerge)ble_location_1/ble_location/cond/MatMul_2&ble_location_1/ble_location/cond/add_1*
N*
T0
�
!ble_location_1/ble_location/add_1Add&ble_location_1/ble_location/cond/Merge'ble_location/ble_location/b_output/read*
T0
\
$ble_location_1/ble_location/IdentityIdentity!ble_location_1/ble_location/add_1*
T0
]
"ble_location_1/ble_location_outputIdentity$ble_location_1/ble_location/Identity*
T0 