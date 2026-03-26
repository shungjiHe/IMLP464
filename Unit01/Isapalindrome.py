def convertInputString():
    # 1. 取得輸入並轉為小寫（英文大小寫視為相同）
    rawInput = input("\nPlease enter a word, phrase, or a sentence \nto check if it is a palindrome: ") 
    rawString = rawInput.lower() 
    # 將字串轉換為 list 方便後續處理
    rawList = list(rawString)
    return rawList

def stripAnalphabetics(dirtyList): 
    analphabeticList = [" ", "-", ".", ",", ":", ";", "!", "?", "'", "\""] 
    # 2. 去除相關標點符號與空白
    cleanedList = []
    for char in dirtyList:
        if char not in analphabeticList:
            cleanedList.append(char)
    return cleanedList

def runPalindromeCheck(straightList):
    # 3. 核心邏輯：利用切片反轉 list
    reversedList = straightList[::-1] 
    if straightList == reversedList: 
        return "The text you have entered is a palindrome!" 
    else: 
        return "The text you have entered is not a palindrome." 

def main(): 
    print("\nPalindrome checker") 
    # 串接所有流程
    original_list = convertInputString()
    clean_list = stripAnalphabetics(original_list)
    result = runPalindromeCheck(clean_list)
    print(result)

if __name__ == "__main__":
    main()