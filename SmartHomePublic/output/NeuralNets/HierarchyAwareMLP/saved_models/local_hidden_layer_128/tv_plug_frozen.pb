
G
input/tv_plugPlaceholder*
dtype0*
shape:���������
9
input/keepprobPlaceholder*
dtype0*
shape:
�
connect_sensor_0/tv_plug/w_0Const*�
value�B�@"�PŽ��>ɶB>�A>5�k>��;�\�C��=�L>�Qͽ�V��p��XU�:O^�>=�0<�܏=��>Q:>C�+���Ā�>7)t����`-�;UCG>�X��.�=�\��F�<�nh� {���2��]�>��k��3����=Õ8>�����m>Ew��c�>i������(<5�=/�	�b�X>�PϽ{�J�c>P.��ʪ=�y�=B�[�(�>yVS>I����9��@g>s�>�lw==��6�>ğ���,��ힱ<훁�Fl>�@>���=e"��T�nl=jʅ����=��=�wB=�8ռr�R�A�=��%���?����>8k����=J���Q`=�oz���>4��!q>�:�=G�s�&3=/�=b伉�����=��G����GV���>�+z>��U���V>��!�D��=�拽��?>�`R=���__==ym�<��K>�����:�cXH>w�E=���=�2���=�=�T���c>��o�S)=��=*
dtype0
�
!connect_sensor_0/tv_plug/w_0/readIdentityconnect_sensor_0/tv_plug/w_0*
T0*/
_class%
#!loc:@connect_sensor_0/tv_plug/w_0
�
connect_sensor_0/tv_plug/bias_0Const*�
value�B�@"����<���=�M�=���=��=<=!�����o <xt�=�lջ�n��b��<I���=�:�H.\=�3�=S�?�� �����<���=�5�<&u��}��;8`�=ճ���c��WP�<^�n��*R�U�Q�
�;�6�=�C<sn�	�=Ct�1lp;�@�=F=��j=��"=.��8L6�7�<䴠�p��=.p^���-<�Ի=��"�d7���:�c&F� �=���=�༠�<3�=�=�B�=0TW;7m�=\��;*
dtype0
�
$connect_sensor_0/tv_plug/bias_0/readIdentityconnect_sensor_0/tv_plug/bias_0*
T0*2
_class(
&$loc:@connect_sensor_0/tv_plug/bias_0
�
!connect_sensor_0_1/tv_plug/MatMulMatMulinput/tv_plug!connect_sensor_0/tv_plug/w_0/read*
transpose_a( *
transpose_b( *
T0
w
connect_sensor_0_1/tv_plug/addAdd!connect_sensor_0_1/tv_plug/MatMul$connect_sensor_0/tv_plug/bias_0/read*
T0
P
connect_sensor_0_1/tv_plug/ReluReluconnect_sensor_0_1/tv_plug/add*
T0
k
(connect_sensor_0_1/tv_plug/dropout/ShapeShapeconnect_sensor_0_1/tv_plug/Relu*
T0*
out_type0
b
5connect_sensor_0_1/tv_plug/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
b
5connect_sensor_0_1/tv_plug/dropout/random_uniform/maxConst*
dtype0*
valueB
 *  �?
�
?connect_sensor_0_1/tv_plug/dropout/random_uniform/RandomUniformRandomUniform(connect_sensor_0_1/tv_plug/dropout/Shape*
T0*
dtype0*
seed2 *

seed 
�
5connect_sensor_0_1/tv_plug/dropout/random_uniform/subSub5connect_sensor_0_1/tv_plug/dropout/random_uniform/max5connect_sensor_0_1/tv_plug/dropout/random_uniform/min*
T0
�
5connect_sensor_0_1/tv_plug/dropout/random_uniform/mulMul?connect_sensor_0_1/tv_plug/dropout/random_uniform/RandomUniform5connect_sensor_0_1/tv_plug/dropout/random_uniform/sub*
T0
�
1connect_sensor_0_1/tv_plug/dropout/random_uniformAdd5connect_sensor_0_1/tv_plug/dropout/random_uniform/mul5connect_sensor_0_1/tv_plug/dropout/random_uniform/min*
T0
y
&connect_sensor_0_1/tv_plug/dropout/addAddinput/keepprob1connect_sensor_0_1/tv_plug/dropout/random_uniform*
T0
b
(connect_sensor_0_1/tv_plug/dropout/FloorFloor&connect_sensor_0_1/tv_plug/dropout/add*
T0
k
&connect_sensor_0_1/tv_plug/dropout/divRealDivconnect_sensor_0_1/tv_plug/Reluinput/keepprob*
T0
�
&connect_sensor_0_1/tv_plug/dropout/mulMul&connect_sensor_0_1/tv_plug/dropout/div(connect_sensor_0_1/tv_plug/dropout/Floor*
T0
�
!connect_sensor_0/tv_plug/w_outputConst*�
value�B�@"��l�<�>��->/�6���v>c�3�&2I>=o�=�>)g��<b��6<�>՟!=��O>����^��>Cث�]W��/�_>��Q>�UK���M�<3-U��ǽ`�2<:�����������]>�Ė=���>p3�ݦ	��]9�!O��E�鼵mY���>M&�>l��=0�ֺC>FȘ;ˤ=�F>�<���џ>�9=8cR>o�}=�Y��F>�ս�x�>%�Z�M���d����=,�>�����X@�fֽw��;k�j��跼�!�=�@>�L=�Kw>H�1�P򽌂���<�Zi>߾�>�z>cE�؞�>g��=����{z���L�P�����5�\�N�=�x:�>b�<�.>S�{�2qn>
ۛ���; �����)&>���=�j��ʲ����l�/>,����hѽ�>\�����>�f"9���>Z���q����f>v*��9>N7�=�逽���>hƲ��"�>C\O��z��]>\��1>t"�:�� d>*
dtype0
�
&connect_sensor_0/tv_plug/w_output/readIdentity!connect_sensor_0/tv_plug/w_output*
T0*4
_class*
(&loc:@connect_sensor_0/tv_plug/w_output
Y
$connect_sensor_0/tv_plug/bias_outputConst*
valueB"�o�=5���*
dtype0
�
)connect_sensor_0/tv_plug/bias_output/readIdentity$connect_sensor_0/tv_plug/bias_output*
T0*7
_class-
+)loc:@connect_sensor_0/tv_plug/bias_output
�
#connect_sensor_0_1/tv_plug_1/MatMulMatMul&connect_sensor_0_1/tv_plug/dropout/mul&connect_sensor_0/tv_plug/w_output/read*
transpose_b( *
T0*
transpose_a( 
�
 connect_sensor_0_1/tv_plug_1/addAdd#connect_sensor_0_1/tv_plug_1/MatMul)connect_sensor_0/tv_plug/bias_output/read*
T0
b
+connect_sensor_0_1/tv_plug_1/tv_plug_outputIdentity connect_sensor_0_1/tv_plug_1/add*
T0 