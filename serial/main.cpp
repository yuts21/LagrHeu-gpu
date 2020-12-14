#include "GAP.h"
#include "Lagrangian.h"
#include <ctime>
#include <iostream>

/******************
 *
 * 需要预先将所有相关数据放入gpu中，方便运算
 * 根据算法复杂度，优化关键步骤即可，无需全部优化
 *
 *******************/

int main(int argc, char *argv[]) {
    // GAP类与对象
    GeneralizedAssignemnt *GAP = new GeneralizedAssignemnt();
    string fileName, path;
    int res;

    bool isVerbose = false;
    string instance = argv[1];
    double alpha = 2.546;
    double alphastep = 0.936; // alpha更新的频率
    double minalpha = 0.01;
    int innerIter = 98; // 每循环x次，更新alpha
    int maxIter = 619;  // 最多循环次数
    int algoType = 2;   // 算法种类，1为松弛容量，其他为松弛分配
    double seed = 505;

    // path = "c:/AAAToBackup/ricerche/GAP/istanze/instances/homemade/";
    path = "";
    fileName = instance;

    if (isVerbose) {
        cout << "alpha=" << alpha << endl;
        cout << "alphastep=" << alphastep << endl;
        cout << "minalpha=" << minalpha << endl;
        cout << "innerIter=" << innerIter << endl;
        cout << "maxiter=" << maxIter << endl;
        cout << "algorithm: " << (algoType == 1 ? "Relax capacities" : "Relax assignments") << endl;
        cout << "verbose: " << (isVerbose ? true : false) << endl;
        cout << "seed = " << seed << endl;
        cout << "instance: " << instance << endl;
        cout << "path: " << path << endl;
    }

    GAP->isVerbose = isVerbose;

    // 读取json格式存储的信息
    /************** 在GeneralizedAssignemnt类与Lagrangian类中加入gpu格式的数组 **********************/
    /************* 读取后，申请并拷贝gpu空间，之后的所有运算均在gpu上进行 *****************************/
    GAP->readData(path + fileName);

    maxIter = maxIter * GAP->n * GAP->m;

    // start_t = clock();
    // printf("%lf\n", start_t);

    // LAGR类与对象
    Lagrangian *LAGR = new Lagrangian(GAP, GAP->zub);

    clock_t start_t = clock();

    if (algoType == 1) {
        if (isVerbose)
            cout << "Relaxing capacities ---------------" << endl;
        res = LAGR->lagrAss(GAP->c, alpha, alphastep, minalpha, innerIter, maxIter);
    } else {
        if (isVerbose)
            cout << "Relaxing assignments ---------------" << endl;
        res = LAGR->lagrCap(GAP->c, alpha, alphastep, minalpha, innerIter, maxIter);
    }
    if (LAGR != NULL)
        delete LAGR;
    LAGR = NULL;

    clock_t end_t = clock();
    cerr << (double)(end_t - start_t) / CLOCKS_PER_SEC << endl;
    if (isVerbose)
        cout << "Time: " << (double)(end_t - start_t) / CLOCKS_PER_SEC << endl;
    cout << GAP->zub << endl;

    if (isVerbose) {
        cout << "\n<ENTER> to exit ...";
        cin.get();
    }
    delete GAP->sol;
    delete GAP->solbest;
    delete GAP;
    return 0;
}
