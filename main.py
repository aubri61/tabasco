from classes import DataManager, Generator


if __name__ == '__main__':

    # init data manager -> read csv and store df
    data_manager = DataManager()
    generator = Generator(data_manager)

    generate_done = generator.generate_dialog()
    # print(data_manager.output[0]['persona'])
    # print('----'*3)
    # print(data_manager.output[0]['gt_movie'])
    # print('----'*3)
    # print(data_manager.output[0]['conversation'])

    # print('\n######################\n')
    # print(data_manager.output[1]['persona'])
    # print('----'*3)
    # print(data_manager.output[1]['conversation'])

    if generate_done == 1:
        print('ok! done :)')
