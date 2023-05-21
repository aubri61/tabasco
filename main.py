from classes import DataManager, Generator


if __name__ == '__main__':
  
  # init data manager -> read csv and store df
  data_manager = DataManager()
  generator = Generator(data_manager)
  # print(data_manager.output[1])
  
  generate_done = generator.generate_dialog()
  
  if generate_done == 1 :
    print('ok! done :)')

  