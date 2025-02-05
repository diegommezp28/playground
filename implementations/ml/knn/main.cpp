#include <stdio.h>
#include "utils.cpp"

using namespace Utils;

int main()
{
    // headers: date,time,username,wrist,activity,acceleration_x,acceleration_y,acceleration_z,gyro_x,gyro_y,gyro_z
    std::vector<ColumnDescriptor> columns = {
        ColumnDescriptor("date", DataType::STRING),
        ColumnDescriptor("time", DataType::STRING),
        ColumnDescriptor("username", DataType::STRING),
        ColumnDescriptor("wrist", DataType::STRING),
        ColumnDescriptor("activity", DataType::STRING),
        ColumnDescriptor("acceleration_x", DataType::DOUBLE),
        ColumnDescriptor("acceleration_y", DataType::DOUBLE),
        ColumnDescriptor("acceleration_z", DataType::DOUBLE),
        ColumnDescriptor("gyro_x", DataType::DOUBLE),
        ColumnDescriptor("gyro_y", DataType::DOUBLE),
        ColumnDescriptor("gyro_z", DataType::DOUBLE)};

    // Reader must be able to do groupby, filter, sort
    Reader reader("../../../datasets/dataset.csv", columns);

    reader.readFile();

    return 0;
}