import pymysql
from pymysql.err import ProgrammingError, MySQLError
from sql_dao import host, port, username, password, database, charset


def get_conn():
    return pymysql.connect(
        host=host,
        port=port,
        user=username,
        password=password,
        database=database,
        charset=charset
    )


insert_polarity_sql = """
    insert into polarity_analy(workId, country, platform, time, positive, negative, neutrality) 
    values({}, "{}", "{}", "{}", {}, {}, {});
"""

insert_sentiment_sql = """
    insert into sentiment_analy(workId, country, platform, time, happy, amazed, neutrality,
    sad, angry, fear) 
    values({}, "{}", "{}", "{}", {}, {}, {}, {}, {}, {});
"""

delete_sentiment_sql = """
    delete from sentiment_analy where workId = {} and country = "{}" and platform = "{}"
    and time = "{}";
"""

query_sentiment_sql = """
    select * from sentiment_analy where workId = {} and country = "{}" and platform = "{}"
    and time = "{}";
"""

delete_polarity_sql = """
    delete from polarity_analy where workId = {} and country = "{}" and platform = "{}"
    and time = "{}";
"""

query_polarity_sql = """
    select * from polarity_analy where workId = {} and country = "{}" and platform = "{}"
    and time = "{}";
"""

# cursor = conn.cursor()
query_sql = """
    select * from user;
"""
# df = pd.read_sql(query_sql, con=conn)
# print(df)

delete_subject_sql = """
    delete from subject_analysis where workId = {} and subject = "{}";
"""

insert_subject_sql = """
    insert into subject_analysis(workId, subject, positive, negative, neutrality) 
    values({}, "{}", {}, {}, {});
"""

delete_comment_subject_sql = """
    delete from comment_subject where commentId = {} and propertyWord = "{}" and opinionWord = "{}";
"""

insert_comment_subject_sql = """
    insert into comment_subject(commentId, workId, propertyWord, opinionWord, sentiment, subjects) 
    values({}, {}, "{}", "{}", "{}", "{}");
"""


def insert_polarity(workId, country, platform, post_time, positive, negative, neutrality, conn):
    cursor = conn.cursor()
    if positive is None:
        positive = 0
    if negative is None:
        negative = 0
    if neutrality is None:
        neutrality = 0
    try:
        cursor.execute(insert_polarity_sql.format(workId, country, platform, post_time, positive, negative, neutrality))
        conn.commit()
        return True
    except ProgrammingError:
        print("sql语法错误")
        return False
    finally:
        cursor.close()


def insert_sentiment(workId, country, platform, post_time, happy, amazed, neutrality, sad, angry, fear, conn):
    cursor = conn.cursor()
    if happy is None:
        happy = 0
    if amazed is None:
        amazed = 0
    if neutrality is None:
        neutrality = 0
    if sad is None:
        sad = 0
    if angry is None:
        angry = 0
    if fear is None:
        fear = 0
    try:
        cursor.execute(insert_sentiment_sql
                       .format(workId, country, platform, post_time, happy, amazed, neutrality, sad, angry, fear))
        conn.commit()
        return True
    except ProgrammingError:
        print("sql语法错误")
        return False
    finally:
        cursor.close()


def insert_subject(workId, subject, positive, negative, neutrality, conn):
    if subject == '其他':
        return
    if positive == 0 and negative == 0 and neutrality == 0:  # 全是0，不插入
        return
    cursor = conn.cursor()
    try:
        cursor.execute(delete_subject_sql.format(workId, subject))  # 先把原来相同的记录删除
        cursor.execute(insert_subject_sql.format(workId, subject, positive, negative, neutrality))
        conn.commit()  # 提交事务
    except ProgrammingError:
        print("sql语法错误")
    except MySQLError:
        print('sql执行错误')
        conn.rollback()
    finally:
        cursor.close()


def insert_comment_subject(commentId, workId, propertyWord, opinionWord, sentiment, subjects, conn):
    cursor = conn.cursor()
    try:
        cursor.execute(delete_comment_subject_sql.format(commentId, propertyWord, opinionWord))  # 先把原来相同的记录删除
        cursor.execute(insert_comment_subject_sql
                       .format(commentId, workId, propertyWord, opinionWord, sentiment, subjects))
        conn.commit()  # 提交事务
    except ProgrammingError:
        print("sql语法错误")
    except MySQLError:
        print('sql执行错误')
        conn.rollback()
    finally:
        cursor.close()


if __name__ == "__main__":
    # conn = get_conn()
    # insert_subject(8, '故事情节', 120, 70, 20, conn)
    pass
