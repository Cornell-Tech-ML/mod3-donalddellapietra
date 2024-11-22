# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.


Here is a brief sanity check that the elementwise multiplication is slower than the real matrix multiplication. See the file test_manual.py for more details.

![MMul Comparison](time_comparison.png)

![Reduce Comparison](reduce_comparison.png)

For numba:
Average time for elementwise multiplication: 1.065640 seconds
Average time for real matrix multiplication: 0.072189 seconds
Average time for reduce: 0.003198 seconds
Average time for reduce numba: 0.022610 seconds (note this is because I only did it 20 times, but as you can see it is faster than the regular version after the first iteration)

For cuda:
Average time for elementwise multiplication: 1.457392 seconds
Average time for cuda matrix multiplication: 0.049690 seconds
Average time for reduce: 0.004651 seconds
Average time for reduce cuda: 0.015866 seconds

SIMPLE DATASET

Epoch  0  loss  6.823333593225619 correct 31
Epoch  10  loss  1.2989138292149742 correct 47
Epoch  20  loss  1.0051613992829407 correct 50
Epoch  30  loss  0.6203572995003972 correct 50
Epoch  40  loss  0.2884476375804359 correct 50
Epoch  50  loss  0.1375977392017274 correct 50
Epoch  60  loss  0.3269712892335858 correct 50
Epoch  70  loss  0.28449133507900826 correct 50
Epoch  80  loss  0.23905051548743667 correct 50
Epoch  90  loss  0.20789659428123958 correct 50
Epoch  100  loss  0.46010060475709347 correct 50
Epoch  110  loss  0.09231967569779649 correct 50
Epoch  120  loss  0.9057859537459119 correct 50
Epoch  130  loss  0.014745552788471616 correct 50
Epoch  140  loss  0.3178900934503781 correct 50
Epoch  150  loss  0.09870086094954413 correct 50
Epoch  160  loss  0.2955135281163285 correct 50
Epoch  170  loss  0.2957327797023588 correct 50
Epoch  180  loss  0.27855880385052273 correct 50
Epoch  190  loss  0.30912139759291446 correct 50
Epoch  200  loss  0.2432269078271828 correct 50
Epoch  210  loss  0.1519864806215312 correct 50
Epoch  220  loss  0.2859499934705349 correct 50
Epoch  230  loss  0.18103385360630095 correct 50
Epoch  240  loss  0.3448996975300639 correct 50
Epoch  250  loss  0.206643567681843 correct 50
Epoch  260  loss  0.21002820063058128 correct 50
Epoch  270  loss  0.5242144911483779 correct 50
Epoch  280  loss  0.2142012000606571 correct 50
Epoch  290  loss  0.03092457664440143 correct 50
Epoch  300  loss  0.07959062298495369 correct 50
Epoch  310  loss  0.050486427181622204 correct 50
Epoch  320  loss  0.041222676459788894 correct 50
Epoch  330  loss  0.005217177666263163 correct 50
Epoch  340  loss  0.004777291624428023 correct 50
Epoch  350  loss  0.33871926974319866 correct 50
Epoch  360  loss  0.029787746201446925 correct 50
Epoch  370  loss  0.002882883722868723 correct 50
Epoch  380  loss  0.11218083456389413 correct 50
Epoch  390  loss  0.0032556650591193475 correct 50
Epoch  400  loss  0.12307464704243101 correct 50
Epoch  410  loss  0.1250823022042029 correct 50
Epoch  420  loss  0.005504797103863354 correct 50
Epoch  430  loss  0.09636159372978823 correct 50
Epoch  440  loss  0.19665241889050694 correct 50
Epoch  450  loss  0.1648799451096511 correct 50
Epoch  460  loss  0.00025790378730714984 correct 50
Epoch  470  loss  0.01145322054925885 correct 50
Epoch  480  loss  0.13781859099756807 correct 50
Epoch  490  loss  0.022754478053070216 correct 50

Average epoch time: 1.673120 seconds


SPLIT DATASET


Epoch  0  loss  7.506276978187855 correct 32
Epoch  10  loss  5.1262630893371055 correct 40
Epoch  20  loss  5.722724201189267 correct 45
Epoch  30  loss  4.224430597723488 correct 46
Epoch  40  loss  2.9459282207563753 correct 47
Epoch  50  loss  1.6782743007299523 correct 46
Epoch  60  loss  4.116299031888678 correct 47
Epoch  70  loss  1.494442988683634 correct 48
Epoch  80  loss  2.8724289169342008 correct 49
Epoch  90  loss  2.55250503072933 correct 48
Epoch  100  loss  1.370647565444746 correct 48
Epoch  110  loss  1.6354420852194813 correct 48
Epoch  120  loss  1.5267183738108563 correct 49
Epoch  130  loss  1.6396870312921892 correct 48
Epoch  140  loss  0.532700523578322 correct 48
Epoch  150  loss  1.0569582570440017 correct 48
Epoch  160  loss  2.0745840853120483 correct 48
Epoch  170  loss  0.1773121810259196 correct 48
Epoch  180  loss  0.5467053640396895 correct 50
Epoch  190  loss  1.1236198774640305 correct 49
Epoch  200  loss  0.5401287692409455 correct 48
Epoch  210  loss  0.9058465277153505 correct 48
Epoch  220  loss  0.528335408575202 correct 49
Epoch  230  loss  0.2897824522432052 correct 48
Epoch  240  loss  1.6647561139697513 correct 50
Epoch  250  loss  1.5830351224869024 correct 50
Epoch  260  loss  0.705208032787609 correct 48
Epoch  270  loss  0.7427509779324821 correct 49
Epoch  280  loss  0.8364580567866163 correct 50
Epoch  290  loss  0.20172985461125012 correct 50
Epoch  300  loss  1.0072478025391052 correct 50
Epoch  310  loss  1.2270484595769577 correct 50
Epoch  320  loss  1.2145931080509262 correct 50
Epoch  330  loss  0.45658549376601026 correct 50
Epoch  340  loss  0.7660193149306838 correct 49
Epoch  350  loss  1.620436216219945 correct 50
Epoch  360  loss  0.8945796215192531 correct 48
Epoch  370  loss  1.213293162973407 correct 50
Epoch  380  loss  2.6347920712250072 correct 45
Epoch  390  loss  0.20192237956160444 correct 50
Epoch  400  loss  1.1328610766944802 correct 50
Epoch  410  loss  1.6407653060013327 correct 50
Epoch  420  loss  0.820309154485144 correct 50
Epoch  430  loss  0.1744059822589843 correct 48
Epoch  440  loss  0.5109811678521681 correct 50
Epoch  450  loss  0.8526636583154583 correct 49
Epoch  460  loss  0.13240624332117096 correct 48
Epoch  470  loss  1.0948191946019654 correct 50
Epoch  480  loss  0.13169688338319843 correct 50
Epoch  490  loss  0.11341494842648718 correct 50

Average epoch time: 1.692832 seconds


XOR DATASET
Epoch  0  loss  5.666955905375837 correct 37
Epoch  10  loss  3.8273439616852736 correct 43
Epoch  20  loss  2.3132040850994384 correct 43
Epoch  30  loss  3.318919321759324 correct 45
Epoch  40  loss  3.5758507344437565 correct 43
Epoch  50  loss  3.2169857526041725 correct 44
Epoch  60  loss  3.2510871457400223 correct 42
Epoch  70  loss  1.7818690014309535 correct 44
Epoch  80  loss  1.3700355219517137 correct 44
Epoch  90  loss  2.5124992975512437 correct 45
Epoch  100  loss  0.8754545743736728 correct 44
Epoch  110  loss  3.8547006704210505 correct 44
Epoch  120  loss  3.687470746529539 correct 45
Epoch  130  loss  2.2962841137735497 correct 44
Epoch  140  loss  3.8321763725665683 correct 45
Epoch  150  loss  1.1299858098450133 correct 46
Epoch  160  loss  1.0452262361348421 correct 46
Epoch  170  loss  2.3769171501606214 correct 46
Epoch  180  loss  2.4768882234124447 correct 45
Epoch  190  loss  0.5948436883401303 correct 47
Epoch  200  loss  0.9360762213315408 correct 46
Epoch  210  loss  2.8516451181436233 correct 46
Epoch  220  loss  3.6925798635958014 correct 46
Epoch  230  loss  1.8495268947650563 correct 46
Epoch  240  loss  3.1912240137280654 correct 45
Epoch  250  loss  1.930883647826511 correct 46
Epoch  260  loss  0.49208680611970723 correct 47
Epoch  270  loss  1.9899840168005924 correct 47
Epoch  280  loss  0.9196856405555793 correct 47
Epoch  290  loss  1.8207671993500882 correct 49
Epoch  300  loss  1.66007088173052 correct 47
Epoch  310  loss  0.8475387119755858 correct 47
Epoch  320  loss  1.8914503334222026 correct 48
Epoch  330  loss  0.6089984474722263 correct 48
Epoch  340  loss  0.9945951272272768 correct 47
Epoch  350  loss  1.790762931823289 correct 49
Epoch  360  loss  1.4928304123287137 correct 48
Epoch  370  loss  1.3077080708580604 correct 47
Epoch  380  loss  0.5521568874657199 correct 48
Epoch  390  loss  1.8471783069631842 correct 48
Epoch  400  loss  1.8322772498295525 correct 48
Epoch  410  loss  0.6497762554966653 correct 49
Epoch  420  loss  1.7121994202192643 correct 48
Epoch  430  loss  1.200855572640437 correct 47
Epoch  440  loss  2.072521193441318 correct 46
Epoch  450  loss  2.610740584959326 correct 50
Epoch  460  loss  0.07349911621837474 correct 46
Epoch  470  loss  1.95892027669926 correct 48
Epoch  480  loss  0.3893945761480294 correct 48
Epoch  490  loss  1.2014547205086752 correct 48
Average epoch time: 1.669622 seconds



CPU XOR DATASET

Epoch  0  loss  8.889360909257697 correct 29
Epoch  10  loss  4.38126248267636 correct 34
Epoch  20  loss  3.5588903342361107 correct 45
Epoch  30  loss  3.522402361064894 correct 44
Epoch  40  loss  3.6717316525248025 correct 45
Epoch  50  loss  3.020360801922475 correct 44
Epoch  60  loss  2.162679914339345 correct 47
Epoch  70  loss  3.4141307652189976 correct 47
Epoch  80  loss  3.2392109339768558 correct 47
Epoch  90  loss  4.175437861320631 correct 47
Epoch  100  loss  3.221134879619675 correct 49
Epoch  110  loss  1.4897949744318741 correct 49
Epoch  120  loss  2.3491764011303147 correct 50
Epoch  130  loss  1.60198377320097 correct 48
Epoch  140  loss  0.9579641452361919 correct 48
Epoch  150  loss  1.6651759058797126 correct 50
Epoch  160  loss  1.2290516850119257 correct 47
Epoch  170  loss  0.5500455068845781 correct 50
Epoch  180  loss  0.8750103952936219 correct 50
Epoch  190  loss  1.0823601733240387 correct 50
Epoch  200  loss  1.5451331445570795 correct 50
Epoch  210  loss  1.3209992976350653 correct 50
Epoch  220  loss  0.40798789833737253 correct 50
Epoch  230  loss  0.987072699216855 correct 50
Epoch  240  loss  0.6208290728474473 correct 50
Epoch  250  loss  1.1801767932578466 correct 50
Epoch  260  loss  1.4674708235960985 correct 50
Epoch  270  loss  0.40572744781450465 correct 50
Epoch  280  loss  1.8930333517181046 correct 50
Epoch  290  loss  1.753810096018687 correct 50
Epoch  300  loss  0.26091822003087134 correct 50
Epoch  310  loss  1.2291948379059245 correct 50
Epoch  320  loss  1.2517609333576225 correct 50
Epoch  330  loss  1.4794282899558697 correct 50
Epoch  340  loss  0.8853390535518226 correct 50
Epoch  350  loss  0.3601800095299867 correct 50
Epoch  360  loss  0.7938594448249385 correct 50
Epoch  370  loss  1.1348351616177481 correct 50
Epoch  380  loss  0.12518800643411032 correct 50
Epoch  390  loss  0.208795333409723 correct 50
Epoch  400  loss  0.7043944680542515 correct 50
Epoch  410  loss  0.49725742759225283 correct 50
Epoch  420  loss  0.45860185659175057 correct 50
Epoch  430  loss  0.027451815672980412 correct 50
Epoch  440  loss  0.11702044489774069 correct 50
Epoch  450  loss  0.5763266782713368 correct 50
Epoch  460  loss  0.4755724428898614 correct 50
Epoch  470  loss  0.7859242222443471 correct 50
Epoch  480  loss  0.2756358972001581 correct 50
Epoch  490  loss  1.0294868583141177 correct 50
Average epoch time: 0.147335 seconds