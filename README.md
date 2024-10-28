# Решение для задачи AIJ Multi-Agent AI
от Голубчика Михаила Игоревича
## Описание
Данный репозиторий содержит реализацию мульти-агентного алгоритма обучения с подкреплением.
Для соревнования: https://dsworks.ru/champ/multiagent-ai

Данное решение показало целевую метрику (Mean Focal Score) до
111. На разных этапах обучения, в том числе и при росте локального скора получался и другой скор в лидерборде, например 106.
Но в целом был выше 100 после достаточно продолжительного обучения (больше 400 - 500 итераций).

## Необходимые ресурсы
Обучение агентов занимало примерно 50 часов на GPU RTX 3090 (процессор intel core i3 десятой серии, pci-e 3.0).
По памяти ресурсов требуется сопоставимо с тем, что было нужно для бейслайн.
## Создание решения

__Шаг 1:__ Установить зависимости при помощи команды ```pip install -r requirements.txt```.
Зависимости те же, что и для бейслайна из недостающего только библиотека: joblib
Так же установлен такой же как был рекомендован torch==2.1.0 и cuda 12.1:
``pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121``

__Шаг 2:__ Поочередно выполнить параграфы ноутбука start_train.ipynb.
Либо запустить скрипт start_train.py

Главным результатом работы будет создание директории `submission_vdn` с обученными агентами.
Так же будет создан архив submission.zip со всем необходимым для сабмита.
В папке data/agents_history будут записаны агенты по состоянию на каждые 10 шагов обучения.

По завершению обучения, в каталоге submission_vdn/agents/agent_0 последний агент заменяется
на лучшего агента во время обучения по результату среднего скора за 10 шагов. То есть в submission_vdn/agents/agent_0,
по итогам обучения, находится не последний обученный агент. Последний обученный агент сохраняется в истории в data/agents_history.
Так же этот лучший агент сохраняется в архив submission.zip

__Шаг 3:__ За графиками обучения можно следить в ноутбуке result.ipynb.
Скользящая средняя установлена на 10 шагов, поэтому до 10-го шага попытка посмотреть графики будет вылетать по ошибке.
При необходимости можно уменьшить window_size

Так же, по команде ```pytest show``` можно увидеть работу агентов на текущем этапе обучения.
Корректно работа агентов при текущих настройках в файле будет только на последних 4-м и 5-м этапах обучения. Так как до этого, обучается только нулевой агент (красный)
И поэтому только нулевой агент будет работать полностью нормально.
