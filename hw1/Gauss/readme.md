Метод Гаусса  
========================
Задание:  
Сравнить решение СЛАУ методом Гаусса на встроенной функции и на своей реализации на случайной матрице с диагональным преобладанием. Провести несколько эксперементов пока время счета меньше 1 сек. Построить графики зависимостей.  
Запуск программы: python3 gauss.py  

Выполнено:  
Реализован метод Гаусса. Ход программы:  
1. Генерируется необходимая матрица-пример с текущим размером.
2. Вызывается функция для решение методом Гаусса. (фиксируется время работы)
3. Вызывается библотечная реализация метода Холецкого. (фиксируется время работы)
4. Сравнение библиотечной реализации и собственной реализации с помощью функции assert
5. Строятся графики, оси подписаны, графики подписаны.