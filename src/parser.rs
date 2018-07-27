use std::num::{ParseFloatError, ParseIntError};

#[derive(Debug, PartialEq)]
struct Token {
    pos: (usize, usize),
    ty: TokenType,
}

impl Token {
    fn remove_pos(self) -> TokenType {
        self.ty
    }
}

#[derive(Debug, PartialEq)]
enum TokenType {
    Identifier(String),
    Int(isize),
    Float(f64),
    Str(String),
    LBracket,
    RBracket,
}

#[derive(Debug, Eq, PartialEq)]
pub struct Tokenizer {
    chars: Vec<char>,
    index: usize,
    pos: (usize, usize),
    prev_pos: (usize, usize),
}

pub type TokenizerResult<T> = Result<T, TokenizerError>;

#[derive(Debug, PartialEq)]
pub enum TokenizerError {
    Str(String),
    Eof,
}

impl From<ParseIntError> for TokenizerError {
    fn from(err: ParseIntError) -> Self {
        TokenizerError::Str(err.to_string())
    }
}

impl From<ParseFloatError> for TokenizerError {
    fn from(err: ParseFloatError) -> Self {
        TokenizerError::Str(err.to_string())
    }
}

impl Tokenizer {
    fn new(input: &str) -> Self {
        Tokenizer {
            chars: input.replace("\r\n", "\n").chars().collect(),
            index: 0,
            pos: (0, 0),
            prev_pos: (0, 0),
        }
    }

    fn tokenize(input: &str) -> TokenizerResult<Vec<Token>> {
        let mut tokenizer = Tokenizer::new(input);
        let mut tokens = vec![];
        while !tokenizer.is_empty() {
            match tokenizer.tokenize_one() {
                Ok(tok) => tokens.push(tok),
                Err(TokenizerError::Eof) => break,
                Err(err) => return Err(err),
            }
        }
        Ok(tokens)
    }

    fn peek(&self) -> TokenizerResult<char> {
        if !self.is_empty() {
            Ok(self.chars[self.index])
        } else {
            Err(TokenizerError::Eof)
        }
    }

    fn next(&mut self) -> TokenizerResult<char> {
        if self.is_empty() {
            return Err(TokenizerError::Eof);
        }
        let ch = self.chars[self.index];
        self.prev_pos = self.pos;
        self.pos = if ch == '\n' {
            (self.pos.0 + 1, 0)
        } else {
            (self.pos.0, self.pos.1 + 1)
        };
        self.index += 1;
        Ok(ch)
    }

    fn is_empty(&self) -> bool {
        self.index >= self.chars.len()
    }

    fn tokenize_one(&mut self) -> TokenizerResult<Token> {
        match self.peek() {
            Ok('#') => {
                while let Ok(ch) = self.next() {
                    if ch == '\n' {
                        break;
                    }
                }
                self.tokenize_one()
            }
            Ok('"') => self.tokenize_str(),
            Ok('[') => {
                let tok = Ok(Token {
                    pos: self.pos,
                    ty: TokenType::LBracket,
                });
                let _ = self.next();
                tok
            }
            Ok(']') => {
                let tok = Ok(Token {
                    pos: self.pos,
                    ty: TokenType::RBracket,
                });
                let _ = self.next();
                tok
            }
            Ok(c) if c == '-' || c == '+' || c.is_numeric() => self.tokenize_num(),
            Ok(c) if c.is_whitespace() => {
                while let Ok(ch) = self.peek() {
                    match ch {
                        ch if ch.is_whitespace() => {
                            self.next()?;
                        }
                        _ => break,
                    }
                }
                self.tokenize_one()
            }
            Ok(c) if c.is_alphabetic() => self.tokenize_id(),
            Ok(c) => Err(TokenizerError::Str(format!(
                "({}, {}) tokenize_one(): unexpected char '{}'",
                self.pos.0, self.pos.1, c,
            ))),
            Err(err) => Err(err),
        }
    }

    fn tokenize_num(&mut self) -> TokenizerResult<Token> {
        let start_pos = self.pos;
        let mut num_chars = vec![];
        match self.peek() {
            Ok(ch @ '-') => {
                num_chars.push(ch);
                let _ = self.next();
            }
            Ok('+') => {
                let _ = self.next();
            }
            _ => (),
        }
        let mut is_float = false;
        loop {
            match self.peek() {
                Ok(ch) if ch.is_numeric() => num_chars.push(self.next()?),
                Ok(ch @ '.') => if is_float {
                    return Err(TokenizerError::Str(format!(
                        "({}, {}) tokenize_num(): unexpected char '{}'",
                        self.pos.0, self.pos.1, ch
                    )));
                } else {
                    is_float = true;
                    num_chars.push(self.next()?);
                },
                Ok('e') => {
                    if num_chars.is_empty() {
                        return Err(TokenizerError::Str(format!(
                            "({}, {}) tokenize_num(): no number",
                            self.pos.0, self.pos.1
                        )));
                    }
                    let _ = self.next();
                    match self.tokenize_num()?.remove_pos() {
                        TokenType::Float(f) => {
                            return Ok(Token {
                                pos: start_pos,
                                ty: TokenType::Float(
                                    num_chars.into_iter().collect::<String>().parse::<f64>()?
                                        * f64::powf(10.0, f),
                                ),
                            })
                        }
                        TokenType::Int(i) if i >= 0 && !is_float => {
                            return Ok(Token {
                                pos: start_pos,
                                ty: TokenType::Int(
                                    num_chars.into_iter().collect::<String>().parse::<isize>()?
                                        * isize::pow(10, i as u32),
                                ),
                            })
                        }
                        TokenType::Int(i) => {
                            return Ok(Token {
                                pos: start_pos,
                                ty: TokenType::Float(
                                    num_chars.into_iter().collect::<String>().parse::<f64>()?
                                        * f64::powf(10.0, i as f64),
                                ),
                            })
                        }
                        _ => {
                            return Err(TokenizerError::Str(format!(
                                "({}, {}) tokenize_num(): no number",
                                start_pos.0, start_pos.1
                            )))
                        }
                    }
                }
                Ok(_) | Err(TokenizerError::Eof) => break,
                Err(err) => return Err(err),
            }
        }
        if num_chars.is_empty() {
            return Err(TokenizerError::Str(format!(
                "({}, {}) tokenize_num(): no number",
                self.pos.0, self.pos.1
            )));
        }
        if is_float {
            Ok(Token {
                pos: start_pos,
                ty: TokenType::Float(num_chars.into_iter().collect::<String>().parse()?),
            })
        } else {
            Ok(Token {
                pos: start_pos,
                ty: TokenType::Int(num_chars.into_iter().collect::<String>().parse()?),
            })
        }
    }

    fn tokenize_str(&mut self) -> TokenizerResult<Token> {
        let start_pos = self.pos;
        // Skip over opening quote
        match self.next() {
            Ok(_) => (),
            Err(TokenizerError::Eof) => {
                return Err(TokenizerError::Str(format!(
                    "({}, {}) tokenize_str(): EOF while processing string",
                    self.pos.0, self.pos.1
                )))
            }
            Err(err) => return Err(err),
        }
        let mut str_chars = vec![];
        loop {
            match self.next() {
                Ok('\\') => match self.next() {
                    Ok('n') => str_chars.push('\n'),
                    Ok(ch) => str_chars.push(ch),
                    Err(TokenizerError::Eof) => {
                        return Err(TokenizerError::Str(format!(
                            "({}, {}) tokenize_str(): EOF while processing escape seq",
                            self.prev_pos.0, self.prev_pos.1
                        )))
                    }
                    Err(err) => return Err(err),
                },
                Ok('"') => {
                    return Ok(Token {
                        pos: start_pos,
                        ty: TokenType::Str(str_chars.into_iter().collect()),
                    });
                }
                Ok(ch) => str_chars.push(ch),
                Err(TokenizerError::Eof) => {
                    return Err(TokenizerError::Str(format!(
                        "({}, {}) tokenize_str(): EOF while processing string",
                        self.pos.0, self.pos.1
                    )))
                }
                Err(err) => return Err(err),
            }
        }
    }

    fn tokenize_id(&mut self) -> TokenizerResult<Token> {
        let start_pos = self.pos;
        let mut id_chars = vec![self.next()?];
        loop {
            match self.peek() {
                Ok(ch) if ch.is_alphanumeric() => id_chars.push(self.next()?),
                Ok(_) | Err(TokenizerError::Eof) => break,
                Err(err) => return Err(err),
            }
        }
        if id_chars.is_empty() {
            return Err(TokenizerError::Str(format!(
                "({}, {}) tokenize_id(): no identifier",
                self.pos.0, self.pos.1
            )));
        }
        Ok(Token {
            pos: start_pos,
            ty: TokenType::Identifier(id_chars.into_iter().collect()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_whitespace() {
        assert_eq!(Tokenizer::tokenize("   "), Ok(vec![]));
    }

    #[test]
    fn test_tokenize_comment() {
        assert_eq!(
            Tokenizer::tokenize("  # comment\n123"),
            Ok(vec![Token {
                pos: (1, 0),
                ty: TokenType::Int(123),
            }])
        );
    }

    #[test]
    fn test_tokenize_positive_int() {
        assert_eq!(
            Tokenizer::tokenize("123"),
            Ok(vec![Token {
                pos: (0, 0),
                ty: TokenType::Int(123),
            }])
        );
    }

    #[test]
    fn test_tokenize_negative_int() {
        assert_eq!(
            Tokenizer::tokenize("-123"),
            Ok(vec![Token {
                pos: (0, 0),
                ty: TokenType::Int(-123),
            }])
        );
    }

    #[test]
    fn test_tokenize_negative_float() {
        assert_eq!(
            Tokenizer::tokenize("-1.23"),
            Ok(vec![Token {
                pos: (0, 0),
                ty: TokenType::Float(-1.23),
            }])
        );
    }

    #[test]
    fn test_tokenize_positive_float() {
        assert_eq!(
            Tokenizer::tokenize("1.23"),
            Ok(vec![Token {
                pos: (0, 0),
                ty: TokenType::Float(1.23),
            }])
        );
    }

    #[test]
    fn test_tokenize_exp1() {
        assert_eq!(
            Tokenizer::tokenize("1.23e12"),
            Ok(vec![Token {
                pos: (0, 0),
                ty: TokenType::Float(1.23 * f64::powf(10.0, 12.0)),
            }])
        );
    }

    #[test]
    fn test_tokenize_exp2() {
        assert_eq!(
            Tokenizer::tokenize("1.23e-12"),
            Ok(vec![Token {
                pos: (0, 0),
                ty: TokenType::Float(1.23 * f64::powf(10.0, -12.0)),
            }])
        );
    }

    #[test]
    fn test_tokenize_exp3() {
        assert_eq!(
            Tokenizer::tokenize("1e12"),
            Ok(vec![Token {
                pos: (0, 0),
                ty: TokenType::Int(isize::pow(10, 12)),
            }])
        );
    }

    #[test]
    fn test_tokenize_exp4() {
        assert_eq!(
            Tokenizer::tokenize("1ed12"),
            Err(TokenizerError::Str(
                "(0, 2) tokenize_num(): no number".to_owned()
            ))
        );
    }

    #[test]
    fn test_tokenize1() {
        assert_eq!(
            Tokenizer::tokenize("Accelerator \"kdtree\" \"float emptybonus\" [0.1]"),
            Ok(vec![
                Token {
                    pos: (0, 0),
                    ty: TokenType::Identifier("Accelerator".to_owned()),
                },
                Token {
                    pos: (0, 12),
                    ty: TokenType::Str("kdtree".to_owned()),
                },
                Token {
                    pos: (0, 21),
                    ty: TokenType::Str("float emptybonus".to_owned()),
                },
                Token {
                    pos: (0, 40),
                    ty: TokenType::LBracket,
                },
                Token {
                    pos: (0, 41),
                    ty: TokenType::Float(0.1),
                },
                Token {
                    pos: (0, 44),
                    ty: TokenType::RBracket,
                },
            ])
        )
    }
}
