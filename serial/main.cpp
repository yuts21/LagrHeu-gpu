#include "GAP.h"
#include "Lagrangian.h"
#include <iostream>

/******************
 *
 * 需要预先将所有相关数据放入gpu中，方便运算
 * 根据算法复杂度，优化关键步骤即可，无需全部优化
 *
 *******************/

int main(int argc, char *argv[]) {
    // 计算运行总时间
    // clock_t start_t = 0, end_t = 1;
    // GAP类与对象
    GeneralizedAssignemnt *GAP = new GeneralizedAssignemnt();
    string fileName, path;
    int res;

    int isVerbose = atoi(argv[2]); // 是否输出调试信息
    string instance = argv[4];     // 数据文件相对路径名（json格式）
    double alpha = atof(argv[6]);
    double alphastep = atof(argv[8]); // alpha更新的频率
    double minalpha = atof(argv[10]);
    int innerIter = atoi(argv[12]); // 每循环x次，更新alpha
    int maxIter = atoi(argv[14]);   // 最多循环次数
    int algoType = atoi(argv[16]);  // 算法种类，1为松弛容量，其他为松弛分配
    double seed = atof(argv[18]);

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

    // end_t = clock();
    // if(isVerbose) cout << "Time: " << (double)(end_t - start_t)/CLOCKS_PER_SEC << endl;
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
