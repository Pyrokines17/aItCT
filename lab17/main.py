import collections
import argparse
import heapq
import os

from bitstring import BitArray

def lz77_encode(text, window_size):
    output = []
    i = 0

    while i < len(text):
        window_start = max(0, i - window_size)
        window = text[window_start:i]
        match_length = 0
        match_offset = 0
        
        for length in range(1, min(256, len(text) - i)):
            substring = text[i:i + length]
            for j in range(len(window) - length + 1):
                if window[j:j + length] == substring:
                    match_length = length
                    match_offset = len(window) - j
        
        if match_length > 0:
            next_char = text[i + match_length] if i + match_length < len(text) else ''
            output.append((match_offset, match_length, next_char))
            i += match_length + (1 if next_char else 0)
        else:
            output.append((0, 0, text[i]))
            i += 1
    
    covered = sum(length + (1 if char else 0) for _, length, char in output)
    print(f"LZ77: {len(output)} троек, покрывают {covered} символов (из {len(text)})")
    
    if covered != len(text):
        print(f"Ошибка: LZ77 покрывает {covered} символов вместо {len(text)}!")
    
    return output

def lz77_decode(encoded):
    output = []

    for offset, length, char in encoded:
        if length == 0:
            output.append(char)
        else:
            start = len(output) - offset
            for i in range(length):
                output.append(output[start + i])
            if char:
                output.append(char)

    return ''.join(output)

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_codes(data):
    freq = collections.Counter(data)

    if not freq:
        return {}, None
    
    heap = [HuffmanNode(char, freq) for char, freq in freq.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(None, left.freq + right.freq)
        parent.left = left
        parent.right = right
        heapq.heappush(heap, parent)
    
    root = heap[0] if heap else None
    codes = {}
    
    def traverse(node, code=''):
        if node.char is not None:
            codes[node.char] = code or '0'
        else:
            if node.left:
                traverse(node.left, code + '0')
            if node.right:
                traverse(node.right, code + '1')
    
    if root:
        traverse(root)

    return codes, root

def huffman_encode(data, codes):
    encoded = BitArray()

    for char in data:
        encoded.append('0b' + codes[char])

    return encoded

def huffman_decode(encoded, root, bit_length, expected_bytes):
    current = root
    output = []

    for i in range(bit_length):
        if i >= len(encoded):
            print(f"Предупреждение: Достигнут конец битового массива на индексе {i}, ожидалось {bit_length} битов")
            break
        if len(output) >= expected_bytes:
            print(f"Декодирование остановлено: Достигнуто ожидаемое количество байтов ({expected_bytes})")
            break

        bit = encoded[i]
        
        if bit:
            current = current.right
        else:
            current = current.left
        if current.char is not None:
            output.append(current.char)
            current = root

    if len(output) != expected_bytes:
        print(f"Ошибка: Декодировано {len(output)} байт, ожидалось {expected_bytes}")
    
    return output

def save_huffman_tree(root):
    if root is None:
        return "0"
    
    if root.char is not None:
        char = chr(root.char) if isinstance(root.char, int) else root.char
        return f"1{char}"
    
    return f"0{save_huffman_tree(root.left)}{save_huffman_tree(root.right)}"

def load_huffman_tree(data, index=0):
    if index >= len(data):
        return None, index
    
    if data[index] == '1':
        char = ord(data[index + 1]) if index + 1 < len(data) else 0
        return HuffmanNode(char, 0), index + 2
    
    node = HuffmanNode(None, 0)
    node.left, index = load_huffman_tree(data, index + 1)
    node.right, index = load_huffman_tree(data, index)

    return node, index

def compress(text, window_size, output_lz77_file, output_lz77_huffman_file):
    lz77_data = lz77_encode(text, window_size)
    lz77_bytes = bytearray()

    for offset, length, char in lz77_data:
        lz77_bytes.extend(offset.to_bytes(2, 'big'))
        lz77_bytes.append(length)
        lz77_bytes.append(ord(char) if char else 0)
    
    with open(output_lz77_file, 'wb') as f:
        f.write(lz77_bytes)
    
    codes, huffman_root = build_huffman_codes(lz77_bytes)

    if not codes:
        print("Данные пусты, сжатие невозможно")
        with open(output_lz77_huffman_file, 'wb') as f:
            f.write(lz77_bytes)
        return lz77_data, BitArray(), {}
    
    huffman_encoded = huffman_encode(lz77_bytes, codes)
    padding_bits = (8 - (len(huffman_encoded) % 8)) % 8

    if padding_bits:
        huffman_encoded.append('0b' + '0' * padding_bits)
    
    huffman_tree = save_huffman_tree(huffman_root)

    with open(output_lz77_huffman_file, 'wb') as f:
        f.write(len(huffman_tree).to_bytes(4, 'big'))
        f.write(huffman_tree.encode('latin1'))
        f.write(len(huffman_encoded).to_bytes(4, 'big'))
        f.write(len(lz77_bytes).to_bytes(4, 'big'))
        f.write(huffman_encoded.bytes)
        actual_written = f.tell()
        print(f"Записано {actual_written} байт в {output_lz77_huffman_file}")
    
    lz77_size = os.path.getsize(output_lz77_file)
    lz77_huffman_size = os.path.getsize(output_lz77_huffman_file)

    print(f"Размер файла (LZ77, окно {window_size}): {lz77_size} байт")
    print(f"Размер файла (LZ77 + Хаффман): {lz77_huffman_size} байт")
    print(f"Размер дерева Хаффмана: {len(huffman_tree)} байт")
    print(f"Длина битового массива Хаффмана: {len(huffman_encoded)} битов")
    print(f"Ожидаемое количество байтов LZ77: {len(lz77_bytes)}")
    
    return lz77_data, huffman_encoded, codes

def decompress(lz77_huffman_file, output_file):
    with open(lz77_huffman_file, 'rb') as f:
        file_data = f.read()
        if len(file_data) < 12:
            print(f"Ошибка: Файл слишком короткий ({len(file_data)} байт)")
            return ""
        
        tree_size = int.from_bytes(file_data[:4], 'big')
        if tree_size > len(file_data) - 12:
            print(f"Ошибка: tree_size ({tree_size}) слишком большой для файла ({len(file_data)} байт)")
            return ""
        
        tree_data = file_data[4:4+tree_size].decode('latin1')
        bit_length = int.from_bytes(file_data[4+tree_size:8+tree_size], 'big')
        expected_bytes = int.from_bytes(file_data[8+tree_size:12+tree_size], 'big')
        encoded_data = BitArray(file_data[12+tree_size:])
    
    print(f"Декодирование: tree_size={tree_size}, bit_length={bit_length}, expected_bytes={expected_bytes}, encoded_data_len={len(encoded_data)}")
    
    if bit_length > len(encoded_data):
        print(f"Ошибка: bit_length ({bit_length}) превышает encoded_data_len ({len(encoded_data)})")
        bit_length = len(encoded_data)
    
    root, _ = load_huffman_tree(tree_data)
    lz77_bytes = huffman_decode(encoded_data, root, bit_length, expected_bytes)
    lz77_bytes = bytearray(lz77_bytes)
    
    print(f"Декодировано {len(lz77_bytes)} байт из Хаффмана")
    
    if len(lz77_bytes) != expected_bytes:
        print(f"Ошибка: Декодировано {len(lz77_bytes)} байт, ожидалось {expected_bytes}")
        return ""
    
    if len(lz77_bytes) % 4 != 0:
        print(f"Ошибка: Длина lz77_bytes ({len(lz77_bytes)}) не кратна 4, ожидаются тройки по 4 байта")
        return ""
    
    lz77_data = []
    i = 0

    while i < len(lz77_bytes):
        if i + 3 >= len(lz77_bytes):
            print(f"Ошибка: Недостаточно байтов для тройки на индексе {i}")
            break

        offset = int.from_bytes(lz77_bytes[i:i+2], 'big')
        length = lz77_bytes[i+2]
        char = chr(lz77_bytes[i+3]) if lz77_bytes[i+3] else ''
        lz77_data.append((offset, length, char))
        i += 4
    
    decoded_text = lz77_decode(lz77_data)
    
    with open(output_file, 'w') as f:
        f.write(decoded_text)
    
    print(f"Длина декодированного текста: {len(decoded_text)}")

    return decoded_text

def process_file(input_file, window_sizes=[512, 1024, 2048]):
    if not os.path.isfile(input_file):
        print(f"Ошибка: Файл {input_file} не существует")
        return
    
    input_dir = os.path.dirname(input_file) or '.'
    filename = os.path.basename(input_file)
    filename_no_ext = os.path.splitext(filename)[0]
    output_dir = os.path.join(input_dir, f"{filename_no_ext}_results")

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Ошибка: Не удалось создать директорию {output_dir}: {e}")
        return
    
    try:
        with open(input_file, 'r') as f:
            text = f.read()
    except IOError as e:
        print(f"Ошибка: Не удалось прочитать файл {input_file}: {e}")
        return
    
    print(f"Длина входного текста: {len(text)} символов")
    
    for window_size in window_sizes:
        print(f"\nТестирование с размером окна {window_size}:")
        
        lz77_file = os.path.join(output_dir, f"lz77_{window_size}.bin")
        lz77_huffman_file = os.path.join(output_dir, f"lz77_huffman_{window_size}.bin")
        decoded_file = os.path.join(output_dir, f"decoded_{window_size}.txt")
        
        lz77_data, huffman_encoded, codes = compress(text, window_size, lz77_file, lz77_huffman_file)
        decoded_text = decompress(lz77_huffman_file, decoded_file)
        
        print(f"Декодирование успешно: {decoded_text == text}")

def main():
    parser = argparse.ArgumentParser(description="Сжатие и декомпрессия файла с использованием LZ77 и Хаффмана")
    parser.add_argument("input_file", type=str, help="Путь до входного файла")
    args = parser.parse_args()
    
    process_file(args.input_file)

if __name__ == "__main__":
    main()
