import client_blur
import client_decrypt
from time import time
import matplotlib.pyplot as plt
import numpy as np
import json

test_folder = './videos/'
test_filename = ['qcrftest_0.mp4', 'qcrftest_1.mp4', 'qcrftest_2.mp4', 'qcrftest_3.mp4', 'qcrftest_4.mp4',
                 'qcrftest_5.mp4', 'qcrftest_6.mp4', 'crftest_0.mp4', 'crftest_1.mp4', 'crftest_2.mp4',
                 'hdtest_0.mp4', 'hdtest_1.mp4', 'hdtest_2.mp4', 'hdtest_3.mp4', "bbt_10s.mp4"]

"""
    test blur (template generate)
"""
# x_axis = [i for i in range(len(test_filename) )]
blur_time = []
encrypt_time = []

for j in range(20):
    for i in range(len(test_filename)):
        start = time()
        client_blur.play(test_filename[i])
        end = time()
        print("video %d: blur time: %f" % (i, end-start))
        blur_time.append(end-start)

    for i in range(len(test_filename)):
        with open('./json/' + test_filename[i] + '.json', 'r') as f:
            data_store = json.load(f)
        start = time()
        client_decrypt.play(test_filename[i], data_store)
        end = time()
        print("video %d: encrypt time: %f" % (i, end-start))
        encrypt_time.append(end-start)
    print("round %s" % j)


x_axis = [i for i in range(len(blur_time))]
plt.plot(x_axis, blur_time, alpha=0.8, label='$blur$')
plt.plot(x_axis, encrypt_time, alpha=0.8, label='$encrypt$')

plt.xlabel('video idx')
plt.ylabel('time to deliver_key/sec')
plt.show()

# start = time()
# encrypt_video.deliver_key(test_folder, test_filename[12])
# print(time()-start)

blur_average = np.mean(blur_time)
blur_max = max(blur_time)
blur_var = np.var(blur_time)
encrypt_average = np.mean(encrypt_time)
encrypt_max = max(encrypt_time)
encrypt_var = np.var(encrypt_time)
overhead_percent = [(encrypt_time[i] - blur_time[i]) / blur_time[i] * 100 for i in range(len(blur_time))]
overhead_sec = [encrypt_time[i] - blur_time[i] for i in range(len(blur_time))]
print(' ')
print("time used:")
print("blur    : avg %s s, max %s s, variance %s" % (blur_average, blur_max, blur_var))
print("encrypt : avg %s s, max %s s, variance %s" % (encrypt_average, encrypt_max, encrypt_var))
print("overhead: avg %s s, max %s s, variance %s" % (np.mean(overhead_sec), max(overhead_sec), np.var(overhead_sec)))
print("oh_perc : avg %s s, max %s s, variance %s" % (np.mean(overhead_percent),
                                                     max(overhead_percent),
                                                     np.var(overhead_percent)))
print("raw data: ")
print(blur_time)
print(encrypt_time)
"""
time used: 15 * 50 test case
time used:
blur    : avg 10.1405899643898 s, max 14.403314113616943 s, variance 6.132842151875927
encrypt : avg 10.062887445290883 s, max 14.223467111587524 s, variance 5.848077741809999
overhead: avg -0.07770251909891765 s, max 0.2826194763183594 s, variance 0.022275360005731184
oh_perc : avg -0.5996766305206859 s, max 2.68200935397235 s, variance 2.1319045732827067
raw data: 
[9.265470504760742, 6.134201526641846, 10.287147045135498, 14.368687629699707, 3.1527111530303955, 10.286678791046143, 12.248502492904663, 10.456292629241943, 10.462731122970581, 10.547033309936523, 11.04245400428772, 11.024846315383911, 11.008397102355957, 10.999142408370972, 10.935445547103882, 9.266979217529297, 6.1301257610321045, 10.286948442459106, 14.365150213241577, 3.1769912242889404, 10.291837215423584, 12.259668588638306, 10.469921112060547, 10.434788703918457, 10.560651779174805, 11.024503231048584, 11.000473499298096, 10.960720539093018, 10.990248441696167, 10.918579578399658, 9.270495653152466, 6.1219162940979, 10.289435863494873, 14.36716365814209, 3.1548380851745605, 10.29515027999878, 12.2678062915802, 10.42714786529541, 10.427113771438599, 10.576261758804321, 10.974747657775879, 10.955954313278198, 10.941807270050049, 11.026167392730713, 10.928312063217163, 9.25364089012146, 6.122099161148071, 10.275205850601196, 14.361028671264648, 3.148059129714966, 10.306559324264526, 12.22604775428772, 10.443521499633789, 10.429993152618408, 10.551762819290161, 10.975007057189941, 11.030386447906494, 10.952464818954468, 10.993678331375122, 10.906064987182617, 9.260888814926147, 6.12556004524231, 10.290949821472168, 14.356751203536987, 3.15730881690979, 10.289923429489136, 12.256593227386475, 10.463656902313232, 10.429414510726929, 10.531960248947144, 10.999850988388062, 11.037486791610718, 10.975698947906494, 10.998999834060669, 10.918130874633789, 9.271855354309082, 6.143344402313232, 10.2764413356781, 14.373839616775513, 3.1525278091430664, 10.28666067123413, 12.268808603286743, 10.460849285125732, 10.445218801498413, 10.547197580337524, 10.990835666656494, 11.009679794311523, 10.972691059112549, 10.98315691947937, 10.923991680145264, 9.279022932052612, 6.126920938491821, 10.281660079956055, 14.369686603546143, 3.1702678203582764, 10.291380405426025, 12.230779886245728, 10.477537155151367, 10.428737878799438, 10.560261726379395, 10.952696800231934, 11.042677640914917, 11.013864517211914, 11.015759706497192, 10.884488821029663, 9.256164789199829, 6.133684396743774, 10.311187267303467, 14.387399435043335, 3.153766632080078, 10.285258293151855, 12.232102155685425, 10.441953420639038, 10.447858572006226, 10.54031777381897, 11.010267972946167, 11.04369592666626, 10.979374885559082, 11.001797914505005, 10.926562309265137, 9.269611358642578, 6.12984037399292, 10.282450437545776, 14.370850801467896, 3.1524016857147217, 10.28425908088684, 12.252944469451904, 10.444241046905518, 10.468069076538086, 10.542901277542114, 11.021004915237427, 11.019744634628296, 11.00214147567749, 11.037349224090576, 10.893958806991577, 9.25435733795166, 6.139373540878296, 10.301259756088257, 14.354352474212646, 3.1529018878936768, 10.282059669494629, 12.25453495979309, 10.463405132293701, 10.446349620819092, 10.544076442718506, 10.970890760421753, 11.01238751411438, 10.98119568824768, 11.035751342773438, 10.87804102897644, 9.268058061599731, 6.121588468551636, 10.306767225265503, 14.369920492172241, 3.154397964477539, 10.291364669799805, 12.238365888595581, 10.45328974723816, 10.446781873703003, 10.559528350830078, 11.019827365875244, 11.04051423072815, 10.998975038528442, 10.994523286819458, 10.875270128250122, 9.267395257949829, 6.114696502685547, 10.276115655899048, 14.366170644760132, 3.1462347507476807, 10.286758661270142, 12.237733364105225, 10.447089672088623, 10.428277492523193, 10.537203788757324, 11.013388872146606, 11.023463487625122, 10.952372312545776, 10.969248533248901, 10.90578556060791, 9.257197141647339, 6.107748031616211, 10.273052215576172, 14.37097454071045, 3.1569721698760986, 10.299712419509888, 12.248143196105957, 10.450448036193848, 10.438356161117554, 10.536797761917114, 11.028014898300171, 11.025731086730957, 10.979782342910767, 10.983942747116089, 10.866819143295288, 9.257795095443726, 6.120823383331299, 10.275965929031372, 14.372079610824585, 3.1444835662841797, 10.297602891921997, 12.252630949020386, 10.437830209732056, 10.445565938949585, 10.544576406478882, 11.01558232307434, 11.032458782196045, 10.957939863204956, 10.950340747833252, 10.908511877059937, 9.272751331329346, 6.123912811279297, 10.279020309448242, 14.37257719039917, 3.1475460529327393, 10.299132585525513, 12.244858026504517, 10.451560497283936, 10.457350492477417, 10.586795568466187, 11.057498931884766, 10.995988845825195, 10.983936548233032, 10.974039316177368, 10.903231143951416, 9.271134614944458, 6.1165101528167725, 10.29503846168518, 14.358062267303467, 3.150644302368164, 10.294176816940308, 12.235106706619263, 10.443435430526733, 10.446665287017822, 10.540844678878784, 11.041233777999878, 11.04256534576416, 10.923217535018921, 11.0291166305542, 10.876268863677979, 9.283290147781372, 6.118046045303345, 10.286342859268188, 14.39266037940979, 3.157660484313965, 10.28000807762146, 12.22967004776001, 10.46339726448059, 10.444459915161133, 10.544698476791382, 11.0465669631958, 11.030532360076904, 11.002209901809692, 10.981239080429077, 10.908421277999878, 9.276831865310669, 6.130446910858154, 10.300037622451782, 14.397649049758911, 3.1516761779785156, 10.296161890029907, 12.247004508972168, 10.472378253936768, 10.45926833152771, 10.560295820236206, 11.008386135101318, 11.030874013900757, 10.957712173461914, 11.000195264816284, 10.909424066543579, 9.267322778701782, 6.125030517578125, 10.30960464477539, 14.391592741012573, 3.161151647567749, 10.284273386001587, 12.222443580627441, 10.46343207359314, 10.448902130126953, 10.558383226394653, 11.009872436523438, 11.04710578918457, 11.045565605163574, 11.008272409439087, 10.896472215652466, 9.265109539031982, 6.119251251220703, 10.291906833648682, 14.403314113616943, 3.1590499877929688, 10.293513298034668, 12.240756273269653, 10.441299200057983, 10.453853607177734, 10.537602186203003, 10.997384786605835, 11.01241421699524, 10.984904050827026, 10.996420860290527, 10.932750463485718]
[9.31430959701538, 6.227632999420166, 10.31901478767395, 14.213144302368164, 3.2117371559143066, 10.200867891311646, 12.18190860748291, 10.325318336486816, 10.355156421661377, 10.814479351043701, 10.855472564697266, 10.761489391326904, 10.913660526275635, 10.60043454170227, 10.817342281341553, 9.29880976676941, 6.2191736698150635, 10.271372556686401, 14.176033973693848, 3.165766954421997, 10.157605409622192, 12.1488196849823, 10.293324708938599, 10.336101055145264, 10.772421836853027, 10.816533088684082, 10.701534986495972, 10.882763147354126, 10.594316959381104, 10.78110671043396, 9.287917137145996, 6.193063497543335, 10.284127712249756, 14.200671434402466, 3.170144557952881, 10.15512728691101, 12.16666579246521, 10.306544542312622, 10.327126502990723, 10.746058702468872, 10.820748567581177, 10.729085922241211, 10.8818998336792, 10.617517948150635, 10.807754278182983, 9.345081567764282, 6.237131118774414, 10.293819189071655, 14.201369285583496, 3.17490816116333, 10.158015727996826, 12.155255317687988, 10.326004981994629, 10.33606505393982, 10.777015209197998, 10.817323684692383, 10.685960292816162, 10.885545253753662, 10.60056447982788, 10.77127194404602, 9.308412551879883, 6.19678258895874, 10.254173994064331, 14.176983833312988, 3.1588149070739746, 10.184341669082642, 12.173582077026367, 10.29038381576538, 10.30875039100647, 10.772448539733887, 10.827379703521729, 10.7334463596344, 10.893005847930908, 10.607471942901611, 10.782597780227661, 9.286656618118286, 6.2065346240997314, 10.270527362823486, 14.18527603149414, 3.1668827533721924, 10.165350198745728, 12.167097806930542, 10.326672554016113, 10.34064245223999, 10.760359048843384, 10.79305648803711, 10.740161895751953, 10.862842798233032, 10.631337642669678, 10.828311204910278, 9.307490587234497, 6.22468113899231, 10.298948287963867, 14.188111066818237, 3.172691583633423, 10.162532567977905, 12.213194608688354, 10.331570148468018, 10.339055299758911, 10.820304870605469, 10.860846281051636, 10.707666397094727, 10.891924858093262, 10.61219596862793, 10.799758434295654, 9.32809329032898, 6.235990524291992, 10.34193229675293, 14.196225881576538, 3.1752288341522217, 10.18657374382019, 12.175275564193726, 10.329350709915161, 10.338613033294678, 10.80083703994751, 10.874656438827515, 10.761540174484253, 10.914403200149536, 10.62958550453186, 10.826461791992188, 9.303464412689209, 6.241085052490234, 10.32555890083313, 14.223467111587524, 3.1838743686676025, 10.172253131866455, 12.207093715667725, 10.328529834747314, 10.37631106376648, 10.799570798873901, 10.845274686813354, 10.775749683380127, 10.905375480651855, 10.597187757492065, 10.812413215637207, 9.308866024017334, 6.215378761291504, 10.317744016647339, 14.195181369781494, 3.1770057678222656, 10.177279949188232, 12.200139045715332, 10.329581499099731, 10.33901071548462, 10.808378458023071, 10.87104868888855, 10.739267349243164, 10.933052778244019, 10.62738585472107, 10.788667440414429, 9.28108024597168, 6.242693185806274, 10.332964897155762, 14.17992615699768, 3.1811916828155518, 10.17125129699707, 12.157351016998291, 10.328125238418579, 10.353407621383667, 10.808671236038208, 10.856204509735107, 10.799017667770386, 10.935376405715942, 10.621009111404419, 10.808992624282837, 9.30681586265564, 6.249088525772095, 10.32367491722107, 14.204440832138062, 3.176943063735962, 10.192593336105347, 12.172242879867554, 10.329088926315308, 10.345964193344116, 10.80644965171814, 10.864543437957764, 10.759258508682251, 10.896814107894897, 10.655769348144531, 10.817902565002441, 9.28801679611206, 6.240640163421631, 10.3272123336792, 14.183856725692749, 3.172736406326294, 10.149484395980835, 12.175220489501953, 10.352344512939453, 10.335545301437378, 10.81704568862915, 10.84241247177124, 10.76802921295166, 10.913938283920288, 10.618364572525024, 10.826148748397827, 9.306876182556152, 6.240440130233765, 10.319181442260742, 14.198310136795044, 3.1648874282836914, 10.16618275642395, 12.203027486801147, 10.318349361419678, 10.34686279296875, 10.769931554794312, 10.848897218704224, 10.725900888442993, 10.873107194900513, 10.623496055603027, 10.77532958984375, 9.292565822601318, 6.207487106323242, 10.277889966964722, 14.211000204086304, 3.172929525375366, 10.161938905715942, 12.171505451202393, 10.300903797149658, 10.326833486557007, 10.78532075881958, 10.850337743759155, 10.743659496307373, 10.870341300964355, 10.6020987033844, 10.807905912399292, 9.329875707626343, 6.239372253417969, 10.318217515945435, 14.202842235565186, 3.1738195419311523, 10.183863401412964, 12.142221450805664, 10.314813137054443, 10.307163953781128, 10.804852724075317, 10.796359777450562, 10.711316585540771, 10.937097549438477, 10.597280502319336, 10.804702997207642, 9.311066627502441, 6.196110248565674, 10.296824216842651, 14.196462869644165, 3.1727707386016846, 10.18264365196228, 12.175493478775024, 10.309896230697632, 10.356541395187378, 10.816256999969482, 10.906389474868774, 10.797105550765991, 10.9186851978302, 10.611219882965088, 10.794870376586914, 9.284121990203857, 6.2376697063446045, 10.338873624801636, 14.21581220626831, 3.174029588699341, 10.197067499160767, 12.204828262329102, 10.341590881347656, 10.34856367111206, 10.817833423614502, 10.882688522338867, 10.75471544265747, 10.93599247932434, 10.635033130645752, 10.790337324142456, 9.349175214767456, 6.2259745597839355, 10.310451030731201, 14.21565580368042, 3.1764087677001953, 10.164439916610718, 12.174070596694946, 10.308616876602173, 10.343286275863647, 10.81604266166687, 10.852773427963257, 10.776428461074829, 10.933044910430908, 10.651159763336182, 10.809207201004028, 9.314901113510132, 6.243065357208252, 10.328455686569214, 14.209662437438965, 3.172945499420166, 10.171748638153076, 12.190193176269531, 10.360421657562256, 10.376443862915039, 10.820221662521362, 10.840532302856445, 10.746093511581421, 10.912861824035645, 10.587067127227783, 10.827581882476807]
"""
