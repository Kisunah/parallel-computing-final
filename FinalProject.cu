// Seth Hanusik
// Parallel Computing
// Final Project

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cmath>
#include <thread>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

using namespace std;

struct Table {
    int stateId;
    int parentStateId;
    int priority;
    int currArr[18];
    char previousMove;
    int totalCost = 0;
    int f;
    int g;
    int h;

    Table();
    Table(int *arr);
};

Table::Table(int *arr) {
    // currArr = arr;
    memcpy(currArr, arr, 18 * sizeof(int));
    priority = 0;
    previousMove = 'X';
    f = 0;
    g = 0;
    h = 0;
}

Table::Table() {
    priority = 0;
    previousMove = 'X';
    f = 0;
    g = 0;
    h = 0;
}

bool checkGoal(Table parentTable, Table goalTable) {
    int* curr = parentTable.currArr;
    int* goal = goalTable.currArr;

    bool equal = true;
    for (int i = 0; i < 18; i++) {
        if (curr[i] != goal[i]) {
            equal = false;
        }
    }

    return equal;
}

int getOpenPosition(Table currTable) {
    int* currVec = currTable.currArr;
    int zeroPosition;

    for (int i = 0; i < 18; i++) {
        if (currVec[i] == 0) {
            zeroPosition = i;
        }
    }

    return zeroPosition;
}

__global__
void generateTables(int openPosition, Table currTable, Table goalTable, int* d_id, Table *childArr, bool *d_left, bool *d_up, bool *d_right, bool *d_down) {
    if (openPosition % 3 != 0 && currTable.previousMove != 'L' && threadIdx.x == 0) {
        int parentCost = currTable.totalCost;
        int movingValue = currTable.currArr[openPosition - 1];
        int *leftArr = (int *)malloc(sizeof(int)*18);
        memcpy(leftArr, currTable.currArr, 18*sizeof(int));
        leftArr[openPosition] = movingValue;
        leftArr[openPosition - 1] = 0;

        Table *leftChild = &childArr[0];
        memcpy(leftChild->currArr, leftArr, 18 * sizeof(int));
        atomicAdd(d_id, 1);
        leftChild->stateId = *d_id;
        leftChild->parentStateId = currTable.stateId;
        leftChild->previousMove = 'R';

        if (movingValue < 7) {
            leftChild->totalCost = parentCost + 1;
        } else if (movingValue < 17) {
            leftChild->totalCost = parentCost + 3;
        } else {
            leftChild->totalCost = parentCost + 15;
        }

        int f = 0;
        int g = leftChild->totalCost;
        int h = 0;

        for (int i = 0; i < 18; i++) {
            if (leftArr[i] != goalTable.currArr[i]) {
                h++;
            }
        }
        f = g + h;
        leftChild->priority = f;

        leftChild->f = f;
        leftChild->g = g;
        leftChild->h = g;

        childArr[0] = *leftChild;
        *d_left = true;
    }

    if (openPosition > 2 && currTable.previousMove != 'U' && threadIdx.x == 1) {
        int parentCost = currTable.totalCost;
        int movingValue = currTable.currArr[openPosition - 3];
        int *upArr = (int *)malloc(sizeof(int)*18);
        memcpy(upArr, currTable.currArr, 18*sizeof(int));
        upArr[openPosition] = movingValue;
        upArr[openPosition - 3] = 0;

        Table *upChild = &childArr[1];
        memcpy(upChild->currArr, upArr, 18* sizeof(18));
        atomicAdd(d_id, 1);
        upChild->stateId = *d_id;
        upChild->parentStateId = currTable.stateId;
        upChild->previousMove = 'U';

        if (movingValue < 7) {
            upChild->totalCost = parentCost + 1;
        } else if (movingValue < 17) {
            upChild->totalCost = parentCost + 3;
        } else {
            upChild->totalCost = parentCost + 15;
        }

        int f = 0;
        int g = upChild->totalCost;
        int h = 0;

        for (int i = 0; i < 18; i++) {
            if (upArr[i] != goalTable.currArr[i]) {
                h++;
            }
        }
        f = g + h;
        upChild->priority = f;

        upChild->f = f;
        upChild->g = g;
        upChild->h = h;

        childArr[1] = *upChild;
        *d_up = true;
    }

    if (openPosition % 3 != 2 && currTable.previousMove != 'R' && threadIdx.x == 2) {
        int parentCost = currTable.totalCost;
        int movingValue = currTable.currArr[openPosition + 1];
        int *rightArr = (int *)malloc(sizeof(int)*18);
        memcpy(rightArr, currTable.currArr, 18*sizeof(int));
        rightArr[openPosition] = movingValue;
        rightArr[openPosition + 1] = 0;

        Table *rightChild = &childArr[2];
        memcpy(rightChild->currArr, rightArr, 18*sizeof(int));
        atomicAdd(d_id, 1);
        rightChild->stateId = *d_id;
        rightChild->parentStateId = currTable.stateId;
        rightChild->previousMove = 'R';

        if (movingValue < 7) {
            rightChild->totalCost = parentCost + 1;
        } else if (movingValue < 17) {
            rightChild->totalCost = parentCost + 3;
        } else {
            rightChild->totalCost = parentCost + 15;
        }

        int f = 0;
        int g = rightChild->totalCost;
        int h = 0;

        for (int i = 0; i < 18; i++) {
            if (rightArr[i] != goalTable.currArr[i]) {
                h++;
            }
        }
        f = g + h;
        rightChild->priority = f;

        rightChild->f = f;
        rightChild->g = g;
        rightChild->h = h;

        childArr[2] = *rightChild;
        *d_right = true;
    }

    if (openPosition < 15 && currTable.previousMove != 'D' && threadIdx.x == 3) {
        int parentCost = currTable.totalCost;
        int movingValue = currTable.currArr[openPosition + 3];
        int *downArr = (int *)malloc(sizeof(int)*18);
        memcpy(downArr, currTable.currArr, 18*sizeof(int));
        downArr[openPosition] = movingValue;
        downArr[openPosition + 3] = 0;


        Table *downChild = &childArr[3];
        memcpy(downChild->currArr, downArr, 18*sizeof(18));
        atomicAdd(d_id, 1);
        downChild->stateId = *d_id;
        downChild->parentStateId = currTable.stateId;
        downChild->previousMove = 'D';

        if (movingValue < 7) {
            downChild->totalCost = parentCost + 1;
        } else if (movingValue < 17) {
            downChild->totalCost = parentCost + 3;
        } else {
            downChild->totalCost = parentCost + 15;
        }

        int f = 0;
        int g = downChild->totalCost;
        int h = 0;

        for (int i = 0; i < 18; i++) {
            if (downArr[i] != goalTable.currArr[i]) {
                h++;
            }
        }
        f = g + h;
        downChild->priority = f;

        downChild->f = f;
        downChild->g = g;
        downChild->h = h;

        childArr[3] = *downChild;
        *d_down = true;
    }

    __syncthreads();
}

void printTable(Table table) {
    int *arr = table.currArr;
    for (int i = 0; i < 18; i++) {
        if (i % 3 == 0 && i != 0) {
            printf("\n");
        }
        printf("%d ", arr[i]);
    }
    printf("\n");
}

thrust::host_vector<Table> sortOpenList(thrust::host_vector<Table> openList) {
    thrust::sort(openList.begin(), openList.end(), [ ](Table lhs, Table rhs) {
        return lhs.priority > rhs.priority;
    });
    return openList;
}

void solve(Table goalTable, int *d_startArr, int *d_goalArr, thrust::host_vector<Table> openList, thrust::host_vector<Table> closedList, int* d_id) {
    bool equal = checkGoal(openList[0], goalTable);
    Table currTable;

    while (!equal) {
        currTable = openList[openList.size() - 1];
        equal = checkGoal(currTable, goalTable);
        if (!equal) {
            openList.pop_back();
            int openPosition = getOpenPosition(currTable);
            int numTablesGenerated = -1;
            int *d_numTablesGenerated;
            cudaMalloc((void**)&d_numTablesGenerated, sizeof(int));
            cudaMemcpy(d_numTablesGenerated, &numTablesGenerated, sizeof(int), cudaMemcpyHostToDevice);
            
            int n = 4;
            Table *childArr, *d_childArr;
            childArr = (Table*)malloc(sizeof(Table)*n);
            for (int i = 0; i < n; i++) {
                int *tempArr = currTable.currArr;
                memcpy(childArr[i].currArr, tempArr, 18 * sizeof(int));

                childArr[i].priority = 0;
                childArr[i].f = 0;
                childArr[i].g = 0;
                childArr[i].h = 0;
            }

            cudaMalloc((void**)&d_childArr, n * sizeof(Table));
            cudaMemcpy(d_childArr, childArr, n * sizeof(Table), cudaMemcpyHostToDevice);

            bool *d_left, *d_up, *d_right, *d_down;
            bool left = false;
            bool up = false;
            bool right = false;
            bool down = false;
            cudaMalloc((void**)&d_left, sizeof(bool));
            cudaMemcpy(d_left, &left, sizeof(bool), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&d_up, sizeof(bool));
            cudaMemcpy(d_up, &up, sizeof(bool), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&d_right, sizeof(bool));
            cudaMemcpy(d_right, &right, sizeof(bool), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&d_down, sizeof(bool));
            cudaMemcpy(d_down, &d_down, sizeof(bool), cudaMemcpyHostToDevice);

            generateTables<<<1,4>>>(openPosition, currTable, goalTable, d_id, d_childArr, d_left, d_up, d_right, d_down);
            cudaDeviceSynchronize();
            cudaMemcpy(childArr, d_childArr, 4 * sizeof(Table), cudaMemcpyDeviceToHost);
            cudaMemcpy(&left, d_left, sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(&up, d_up, sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(&right, d_right, sizeof(bool), cudaMemcpyDeviceToHost);
            cudaMemcpy(&down, d_down, sizeof(bool), cudaMemcpyDeviceToHost);

            if (left) {
                openList.push_back(childArr[0]);
            }
            if (up) {
                openList.push_back(childArr[1]);
            }
            if (right) {
                openList.push_back(childArr[2]);
            }
            if (down) {
                openList.push_back(childArr[3]);
            }
            openList = sortOpenList(openList);
            closedList.push_back(currTable);
        }
    }
    closedList.push_back(currTable);
}

int main() {
    thrust::host_vector<int> startVec(18);
    thrust::host_vector<int> goalVec(18);

    startVec[0] = 3;
    startVec[1] = 1;
    startVec[2] = 2;
    startVec[3] = 6;
    startVec[4] = 4;
    startVec[5] = 5;
    startVec[6] = 9;
    startVec[7] = 7;
    startVec[8] = 8;
    startVec[9] = 10;
    startVec[10] = 0;
    startVec[11] = 11;
    startVec[12] = 12;
    startVec[13] = 13;
    startVec[14] = 14;
    startVec[15] = 15;
    startVec[16] = 16;
    startVec[17] = 17;

    goalVec[0] = 0;
    goalVec[1] = 1;
    goalVec[2] = 2;
    goalVec[3] = 3;
    goalVec[4] = 4;
    goalVec[5] = 5;
    goalVec[6] = 6;
    goalVec[7] = 7;
    goalVec[8] = 8;
    goalVec[9] = 9;
    goalVec[10] = 10;
    goalVec[11] = 11;
    goalVec[12] = 12;
    goalVec[13] = 13;
    goalVec[14] = 14;
    goalVec[15] = 15;
    goalVec[16] = 16;
    goalVec[17] = 17;

    thrust::device_vector<int> d_startVec = startVec;
    thrust::device_vector<int> d_goalVec = goalVec;

    int* startArr = thrust::raw_pointer_cast(&startVec[0]);
    int* goalArr = thrust::raw_pointer_cast(&goalVec[0]);
    int* d_startArr = thrust::raw_pointer_cast(&d_startVec[0]);
    int* d_goalArr = thrust::raw_pointer_cast(&d_goalVec[0]);

    Table startTable = Table(startArr);
    startTable.stateId = 1;
    Table goalTable = Table(goalArr);
    Table *d_startTable, *d_goalTable;
    cudaMemcpy(d_startTable, &startTable, sizeof(Table), cudaMemcpyHostToDevice);
    cudaMemcpy(d_goalTable, &goalTable, sizeof(Table), cudaMemcpyHostToDevice);

    thrust::host_vector<Table> openList(0);
    thrust::host_vector<Table> closedList(0);
    openList.push_back(startTable);

    int id = 1;
    int *d_id;
    cudaMalloc((void**)&d_id, sizeof(int));
    cudaMemcpy(d_id, &id, sizeof(int), cudaMemcpyHostToDevice);

    auto start = std::chrono::steady_clock::now();

    solve(goalTable, d_startArr, d_goalArr, openList, closedList, d_id);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    cout << "Time: " << elapsed_seconds.count() << endl;
    return 0;
}