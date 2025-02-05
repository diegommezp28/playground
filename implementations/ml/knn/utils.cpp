// includes
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>

// namespace
namespace Utils
{
    enum class DataType
    {
        INT,
        DOUBLE,
        UINT,
        STRING,
        BOOL
    };

    class ColumnDescriptor
    {
    public:
        ColumnDescriptor(const std::string &name, const DataType &type)
        {
            this->name = name;
            this->type = type;
        }

        std::string getName() const
        {
            return name;
        }

        DataType getType() const
        {
            return type;
        }

    private:
        std::string name;
        DataType type;
    };

    class Reader
    {
    public:
        Reader(const std::string &filename, std::vector<ColumnDescriptor> &columns)
        {
            this->filename = filename;
            this->columns = columns;
        }

        /**
         * @brief Reads data from a file specified by the member variable \c filename.
         *
         * Opens the file and reads the first line, parsing comma-separated values into
         * doubles. Each parsed row is stored in the \c read vector.
         *
         * @throws std::runtime_error If the file cannot be opened.
         * @throws std::invalid_argument If any non-numerical value is encountered during parsing.
         */
        void readFile()
        {
            std::ifstream file(filename);

            std::cout << "Trying to read filename: " << filename << std::endl;

            if (!file.is_open())
            {
                throw std::runtime_error("Could not open file");
            }

            std::string line;
            bool headers = true;
            while (std::getline(file, line))
            {
                std::stringstream ss(line);
                std::string value;
                std::vector<std::string> row;

                while (std::getline(ss, value, ','))
                {
                    try
                    {
                        if (!headers)
                        {
                            row.push_back(value);
                        }
                    }
                    catch (const std::invalid_argument &e)
                    {
                        std::cerr << "Invalid number in file: " << value << std::endl;
                        throw;
                    }
                }

                data.push_back(row);

                if (headers)
                {
                    headers = false;
                }
            }

            file.close();
        }

        std::string getFilename() const
        {
            return filename;
        }

        std::vector<ColumnDescriptor> getColumns() const
        {
            return columns;
        }

        std::vector<std::vector<std::string>> getData() const
        {
            return data;
        }

    private:
        std::string filename;
        std::vector<ColumnDescriptor> columns{};
        std::vector<std::vector<std::string>> data{};
    };

}