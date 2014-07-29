#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <iostream>
using namespace std;

//Источник = фичи + пулл
struct TInstance {
    vector<float> Features;//Фичи
    float Goal;//Результат
public:
	    //Конструктор
        TInstance(const string& descr) {
        ParseFromString(descr);
    }

		//Перемешиваем фичи
	    vector<float> Split(const string& descr) {
        size_t begin = 0;
        size_t end = 0;
        while (begin < descr.length()) {
            while (descr[end] != '\t' && end < descr.length()) {
                ++end;
            }
			float value = atof(string(descr.begin() + begin, descr.begin() + end).c_str());
            begin = ++end;
			Features.push_back(value);
        }

		return Features;
    }
	//Считываем инсточники со строки
    void ParseFromString(const string& descr) {
        vector<float> values = Split(descr);
        Goal = values[1];
        Features = vector<float>(values.begin() + 4, values.end());
    }
};

//Пулл
struct TPool {
    vector<TInstance> Instances;//Источники

	//Считываем источники из фала
	void ReadFromFile(const string& filename) {
        fstream featuresIn(filename, std::ios::in);
        char ch[10000];
        while (!featuresIn.eof()) {
            featuresIn.getline(ch, 10000);
            string descr(ch);
            if (!descr.empty()) {
              Instances.push_back(TInstance(string(ch)));
            }
        }
    }
	
};

//Предиктор
class TPredictor {
private:
    size_t FeatureNumber;//Кол-во фич
     
	//y = a*x + b
    float Factor;//Фактор = a
    float Offset;//Оффсет = b

	
public:
	 //гол = фича*фактор+оффсет
	 float Prediction(const vector<float>& features) const {
        return Factor * features[FeatureNumber] + Offset;
    }
    
	//Метрика = среднеквадратичное отклонение
	static float Metric(const TPredictor& predictor, const TPool& pool) {
    float sumSquaredErrors = 0.f;
    for (size_t i = 0; i < pool.Instances.size(); ++i) {
        float prediction = predictor.Prediction(pool.Instances[i].Features);
        float goal = pool.Instances[i].Goal;
        sumSquaredErrors += (prediction - goal) * (prediction - goal);
    }
    return sqrt(sumSquaredErrors / pool.Instances.size());
    }

	//Обучаем машину
	void Learn(const TPool& pool) {
		vector<float> Errors;//Ошибки вычисления
		vector<float> Factors;//Факторы
		vector<float> Offsets;//Оффсеты
		for (size_t featureNumber = 0; featureNumber < pool.Instances[0].Features.size(); ++featureNumber) {
            // определяем оптимальные factor, offset для фичи с номером featureNumber
            // посчитали ошибку на пуле 
			float featureFactor=0.f;
			float featureOffset=0.f;
			float featureSummX=0.f;
			float featureSummY=0.f;
			float avgX=0.f;
			float avgY=0.f;
			for(size_t i=0;i<pool.Instances.size();i++)
			{
				avgY += pool.Instances[i].Goal;
				avgX += pool.Instances[i].Features[featureNumber];
			    	
			}
			avgX/=pool.Instances.size();
			avgY/=pool.Instances.size();
			for(size_t i=0;i<pool.Instances.size();i++)
			{
			    featureSummX+=(pool.Instances[i].Features[featureNumber]-avgX)*(pool.Instances[i].Features[featureNumber]-avgX);
				featureSummY+=(pool.Instances[i].Features[featureNumber]-avgX)*(pool.Instances[i].Goal-avgY);	
			}
			featureFactor = featureSummY/featureSummX;
			featureOffset = -featureFactor*avgX+avgY;
			Factor=featureFactor;
			Offset=featureOffset;
			FeatureNumber = featureNumber;
			float Error = Metric(*this,pool);
			Errors.push_back(Error);
			Factors.push_back(featureFactor);
			Offsets.push_back(featureOffset);
			std::cout << Error<<std::endl;
        }
		
		//высчитываем минимум ошибки
		float minError = Errors[0]; 
        for (size_t featureNumber = 0; featureNumber < pool.Instances[0].Features.size(); ++featureNumber) {
		   if(Errors[featureNumber]<minError)
		   {
			  Factor=Factors[featureNumber];
			  Offset = Offsets[featureNumber];
			  minError = Errors[featureNumber];
			  FeatureNumber = featureNumber;
		   }
		}

		return;
    }
};


//Перетасовывааем фичи
void Shuffle(vector<size_t>& vector) {
	int length = rand();
	for(size_t j=0; j<length;j++)
	{
	 for (size_t i = 1; i < vector.size(); ++i) {
         size_t number = rand() % i;
         swap(vector[number], vector[i]);
     }
	}
}

//Скользящий контроль = тусуем обучающую и тестовые выборки, выбмраем оптимальный вариант
float CrossValidation(TPredictor& predictor, const TPool& pool, size_t foldsCount) {
    vector<TPool> learnFolds(foldsCount);
    vector<TPool> testFolds(foldsCount);

    vector<size_t> instanceNumbers;
    for (size_t i = 0; i < pool.Instances.size(); ++i) {
        instanceNumbers.push_back(i);
    }
    Shuffle(instanceNumbers);
    
    for (size_t i = 0; i < pool.Instances.size(); ++i) {
      for (size_t j = 0; j < foldsCount; ++j) {
            (j == i % foldsCount ? testFolds[j] : learnFolds[j]).Instances.push_back(pool.Instances[instanceNumbers[i]]);
        }
    }

    float avgMetric = 0.f;
    for (size_t i = 0; i < foldsCount; ++i) {
        predictor.Learn(learnFolds[i]);
        avgMetric += predictor.Metric(predictor, testFolds[i]);
    }
    return avgMetric / foldsCount;
}

int main() {
	//Тестовое обучение на основе линейной регрессии
	TPool pool;
	pool.ReadFromFile("machine_cpu.features");
    TPredictor predictor;
    float cv=0;
	size_t index=0;
	//Запускаем скользящий контроль
	do
	{
	   srand(index);
	   float cv=CrossValidation(predictor,pool,10);
	   cout<<"Cross Validation: "<<cv<<"		index: "<<index<<endl;
	   index++;
	}
	while(abs(100-cv)>=5);//Условие выхода, результат не отличается от результата "вики" на 5
}




