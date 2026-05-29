
def findSyntenyReal1(genome1, genome2):
    genome1size = len(genome1)
    genome2size = len(genome2)
    # block = []
    blocks = []

    # Because the genomes are circular, move the last gene in the first genome to position 0 until it is no longer the start of a block
    push = 0
    j = 0
    while j < genome2size:
        if push < min(genome1size,genome2size) and j > -1 and genome2[j] == genome1[-1] and genome2[(j+1)%genome2size] == genome1[0]:
            if push == 0:
                startPos2 = (j+1)%genome2size
            push += 1
            genome1.insert(0, genome1.pop())
            j -= 2
        j += 1

    # If the genomes are the same, return early
    if min(genome1size,genome2size) == push:
        blocks.append([push,0,genome1size-1,startPos2,(startPos2-1)%genome2size])
        return blocks

    # Iterate through the genomes to find synteny blocks
    blocksTemp = []
    addedBlock = True
    while addedBlock:
        addedBlock = False
        i = 0
        while i < genome1size:
            length = 0
            longestBlockLength = 0

            # Skip deleted genes
            while (i < genome1size and genome1[i] == -1):
                i += 1
            if (i == genome1size): # reached end of genome
                break
            
            # Compare with second genome (regular form)
            for j in range(genome2size):
                # First gene in block found, find block length
                if i < genome1size and genome1[i] != -1 and genome1[i] == genome2[j]:
                    length = 1
                    while (genome1[(i+length)%genome1size] != -1 and genome1[(i+length)%genome1size] == genome2[(j+length)%genome2size]):
                        length += 1

                    # If the current block is longer than the previous longest block, replace it
                    if (length > longestBlockLength):
                        inversed = False
                        longestBlockLength = length
                        startPos1 = i
                        startPos2 = j
                    j = j+length
            
            # Compare with second genome (inversed form)
            for j in range(genome2size-1, 0, -1):
                # First gene in block found, find block length
                if i < genome1size and genome1[i] != -1 and genome1[i] == genome2[j]:
                    length = 1
                    while (length < genome1size and genome1[(i+length)%genome1size] != -1 and genome1[(i+length)%genome1size] == genome2[(j-length)%genome2size]):
                        length += 1

                    # If the current block is longer than the previous longest block, replace it
                    if (length > longestBlockLength):
                        inversed = True
                        longestBlockLength = length
                        startPos1 = i
                        startPos2 = j
                    j = j-length

            # Add the longest block found to the list of blocks
            if (longestBlockLength > 0):
                addedBlock = True
                if (inversed):
                    blocksTemp.append([(startPos1-push)%genome1size, (startPos1+longestBlockLength-push-1),
                                        startPos2, (startPos2-longestBlockLength+1), -longestBlockLength])
                else:
                    blocksTemp.append([(startPos1-push)%genome1size, (startPos1+longestBlockLength-push-1),
                                        startPos2, (startPos2+longestBlockLength-1), longestBlockLength])
                i += longestBlockLength
            # If no block found, delete the gene (because no block will ever be found for it)
            else:
                genome1[i] = -1
                i += 1

        if addedBlock:
            blocksTemp = sorted(blocksTemp, key=lambda x: abs(x[0]), reverse=True)
            clearOverlaps(blocksTemp)
            deleteGenes(genome1, genome2, blocksTemp)
            # Move all the new blocks to the list of all blocks
            while len(blocksTemp) != 0:
                blocks.append(blocksTemp.pop(0))
        
    blocks = sorted(blocks, key=lambda x: abs(x[4]), reverse=True)

    # Increase all indexes by 1 to start at 1 instead of 0
    for i in range(len(blocks)):
        blocks[i][0] += 1
        blocks[i][1] += 1
        blocks[i][2] += 1
        blocks[i][3] += 1
        blocks[i][4] = abs(blocks[i][4])
    
    return blocks

# Remove any overlapping blocks
def clearOverlaps(blocks):
    i = 0
    while i < len(blocks):
        j = i+1
        while j < len(blocks):
            # Block j begins after block i and ends before block i or it ends after block i begins and ends before i ends
            if doesOverlap(blocks[i][0],blocks[i][1],blocks[j][0],blocks[j][1]) or \
               doesOverlap(blocks[i][2],blocks[i][3],blocks[j][2],blocks[j][3]):
                del blocks[j]
            else:
                j += 1
        i += 1

# Check if there is overlap between blocks i and j
def doesOverlap(starti, endi, startj, endj):
    return (starti <= max(startj,endj)) and (endi >= min(startj,endj))

    # # If the block is inversed
    # if jIsInversed:
    #     temp = startj
    #     startj = endj
    #     endj = temp

    # if iIsInversed:
    #     temp = starti
    #     starti = endi
    #     endi = temp 

    # # Both blocks wrap around (therefore both must overlap at 0)
    # if starti > endi and startj > endj:
    #     return True

    # # Circular case where only block i wraps around
    # if starti > endi:
    #     return doesOverlap(starti,genomesize-1,startj,endj,genomesize) or doesOverlap(0,endi,startj,endj,genomesize)
    
    # # Circular case where only block j wraps around
    # if startj > endj:
    #     return doesOverlap(starti,endi,startj,genomesize-1,genomesize) or doesOverlap(starti,endi,0,endj,genomesize)
    
    # # Regular case (no wrapping around)
    # return (starti <= endj) and (endi >= startj)

# Delete all genes found in blocks from the genomes
def deleteGenes(genome1, genome2, blocks):
    for i in range(len(blocks)):
        for j in range(blocks[i][0],blocks[i][1]+1):
            genome1[j%len(genome1)] = -1
        for j in range(min(blocks[i][2],blocks[i][3]),max(blocks[i][2],blocks[i][3])+1):
            genome2[j%len(genome2)] = -1

# Convert block count to frequency count
def normalizeMatrix(blockCount):
    # Delete useless entries
    while (blockCount[-1] == 0):
        blockCount.pop()

    # Calculate the total amount of blocks
    totalBlocks = 0
    for i in range(len(blockCount)):
        totalBlocks += blockCount[i]
    #print(totalBlocks)

    if totalBlocks > 0:
        for i in range(len(blockCount)):
            blockCount[i] /= totalBlocks


#slightly edited version of findSyntenyJump 
def findSimBlocks(genome1, genome2):
    GENOME_SIZE = len(genome1)
    block_lengths = []

    for i in range(GENOME_SIZE):
        length = 0
        longestBlockLength = 0

        while i < GENOME_SIZE and genome1[i] == -1:
            i += 1
        if i == GENOME_SIZE:
            break

        for j in range(GENOME_SIZE):
            if i < GENOME_SIZE and genome1[i] != -1 and genome1[i] == genome2[j]:
                length = 1
                while (i + length < GENOME_SIZE and j + length < GENOME_SIZE and
                       genome1[i + length] != -1 and genome1[i + length] == genome2[j + length]):
                    length += 1

                if length > longestBlockLength:
                    longestBlockLength = length
                    startPos = j
                j = j + length

        if longestBlockLength > 0:
            for k in range(longestBlockLength):
                genome1[i + k] = -1
                genome2[startPos + k] = -1
            block_lengths.append(longestBlockLength)

    return block_lengths

def findSyntenyReal2(genome1, genome2):
    genome1size = len(genome1)
    genome2size = len(genome2)
    max_block_len = min(genome1size, genome2size)
    # block = []
    blocks = []

    # Because the genomes are circular, move the last gene in the first genome to position 0 until it is no longer the start of a block
    push = 0
    j = 0
    while j < genome2size:
        if push < min(genome1size,genome2size) and j > -1 and genome2[j] == genome1[-1] and genome2[(j+1)%genome2size] == genome1[0]:
            if push == 0:
                startPos2 = (j+1)%genome2size
            push += 1
            genome1.insert(0, genome1.pop())
            j -= 2
        j += 1

    # If the genomes are the same, return early
    if min(genome1size,genome2size) == push:
        blocks.append([push,0,genome1size-1,startPos2,(startPos2-1)%genome2size])
        return blocks

    # Iterate through the genomes to find synteny blocks
    blocksTemp = []
    addedBlock = True
    while addedBlock:
        addedBlock = False

        # Build an index: gene -> sorted list of positions in genome2 (excluding deleted)
        pos2 = {}
        for jj in range(genome2size):
            g = genome2[jj]
            if g != -1:
                lst = pos2.get(g)
                if lst is None:
                    pos2[g] = [jj]
                else:
                    lst.append(jj)

        i = 0
        while i < genome1size:
            longestBlockLength = 0
            # Ensure defined each i-iteration
            inversed = False

            # Skip deleted genes
            while i < genome1size and genome1[i] == -1:
                i += 1
            if i == genome1size:  # reached end of genome
                break

            gene = genome1[i]

            # ---- Forward (regular) matches: iterate only candidate js ----
            cand_js = pos2.get(gene, ())
            # Iterate in ascending j to preserve your original tie behavior (first-found kept)
            for j in cand_js:
                length = 1
                # extend forward
                while (length < max_block_len and
                    genome1[(i+length) % genome1size] != -1 and
                    genome1[(i+length) % genome1size] == genome2[(j+length) % genome2size]):
                    length += 1
                if length > longestBlockLength:
                    inversed = False
                    longestBlockLength = length
                    startPos1 = i
                    startPos2 = j

            # ---- Inverse matches: same candidate js but scan in descending order ----
            # Keep j > 0 to exactly preserve your current behavior that skips index 0
            for j in reversed(cand_js):
                if j == 0:
                    continue
                length = 1
                # extend backward on genome2
                while (length < max_block_len and
                    genome1[(i+length) % genome1size] != -1 and
                    genome1[(i+length) % genome1size] == genome2[(j-length) % genome2size]):
                    length += 1
                if length > longestBlockLength:
                    inversed = True
                    longestBlockLength = length
                    startPos1 = i
                    startPos2 = j
            # Add the longest block found to the list of blocks
            if (longestBlockLength > 0):
                addedBlock = True
                if (inversed):
                    blocksTemp.append([(startPos1-push)%genome1size, (startPos1+longestBlockLength-push-1),
                                        startPos2, (startPos2-longestBlockLength+1), -longestBlockLength])
                else:
                    blocksTemp.append([(startPos1-push)%genome1size, (startPos1+longestBlockLength-push-1),
                                        startPos2, (startPos2+longestBlockLength-1), longestBlockLength])
                i += longestBlockLength
            # If no block found, delete the gene (because no block will ever be found for it)
            else:
                genome1[i] = -1
                i += 1

        if addedBlock:
            blocksTemp = sorted(blocksTemp, key=lambda x: abs(x[0]), reverse=True)
            clearOverlaps(blocksTemp)
            deleteGenes(genome1, genome2, blocksTemp)
            # Move all the new blocks to the list of all blocks
            while len(blocksTemp) != 0:
                blocks.append(blocksTemp.pop(0))
        
    blocks = sorted(blocks, key=lambda x: abs(x[4]), reverse=True)

    # Increase all indexes by 1 to start at 1 instead of 0
    for i in range(len(blocks)):
        blocks[i][0] += 1
        blocks[i][1] += 1
        blocks[i][2] += 1
        blocks[i][3] += 1
        blocks[i][4] = abs(blocks[i][4])
    
    return blocks


def findSyntenyMulti(genome1_parts, genome2_parts, min_len=2, prefilter=True):
    """
    genome*_parts: list of chromosomes; each chromosome is a list[gene_id] (circular).
    Returns rows: [chr1_idx, L1, R1, chr2_idx, L2, R2, length] (1-based).
    """
    out = []
    for c1, chr1 in enumerate(genome1_parts, start=1):
        if not chr1: 
            continue
        s1 = set(chr1) if prefilter else None
        for c2, chr2 in enumerate(genome2_parts, start=1):
            if not chr2:
                continue
            if prefilter and s1.isdisjoint(chr2):
                continue
            # IMPORTANT: pass copies because your function mutates inputs
            blocks = findSyntenyReal2(chr1[:], chr2[:])
            for L1, R1, L2, R2, K in blocks:
                if K >= min_len:
                    out.append([c1, L1, R1, c2, L2, R2, K])
    out.sort(key=lambda r: r[-1], reverse=True)
    return out
